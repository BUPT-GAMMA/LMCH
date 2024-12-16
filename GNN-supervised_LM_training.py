from utils import *
from parser_args import parser_args
from LM_GNN_trainer import LM_Trainer, GNN_Trainer


def main(args):
    
    for seed in list(map(int, args.seeds.strip().split(','))):

        experiment_name = f"three_wo_dblp_metapath_seed_{seed}"
        data_dir = "./"

        seed_setting(seed)
        
        infer_filepath = data_dir + "infer/"
        LM_prt_ckpt_filepath = data_dir + experiment_name + "/checkpoints/LM_fine_tuning/"
        LM_ckpt_filepath = data_dir + experiment_name + "/checkpoints/LM/"
        GNN_ckpt_filepath = data_dir + experiment_name + "/checkpoints/GNN/"
        LM_intermediate_data_filepath = data_dir + experiment_name + "/intermediate/LM/"
        GNN_intermediate_data_filepath = data_dir + experiment_name + "/intermediate/GNN/"

        # few-shot learning
        # 3-way 3-shot
        dataset_filepath = "metapath_corpus.txt"
        label_filepath = "labels.pkl"
        data = load_few_shot_data(target_dataset='dblp',
                                  dataset_filepath=dataset_filepath, 
                                  label_filepath=label_filepath, 
                                  use_GNN=args.use_GNN)

        run = setup_wandb(args, experiment_name, seed)

        LMTrainer = LM_Trainer(
            output_size=args.LM_output_size,
            classifier_n_layers=args.LM_classifier_n_layers,
            classifier_hidden_dim=args.LM_classifier_hidden_dim,
            device=args.device,
            pretrain_epochs=args.LM_pretrain_epochs,
            optimizer_name=args.optimizer_LM,
            lr=args.lr_LM,
            weight_decay=args.weight_decay_LM,
            dropout=args.dropout,
            att_dropout=args.LM_att_dropout,
            lm_dropout=args.LM_dropout,
            warmup=args.warmup,
            label_smoothing_factor=args.label_smoothing_factor,
            pl_weight=args.alpha,
            max_length=args.max_length,
            batch_size=args.batch_size_LM,
            grad_accumulation=args.LM_accumulation,
            lm_epochs_per_iter=args.LM_epochs_per_iter,
            temperature=args.temperature,
            pl_ratio=args.pl_ratio_LM,
            intermediate_data_filepath=LM_intermediate_data_filepath,
            ckpt_filepath=LM_ckpt_filepath,
            pretrain_ckpt_filepath=LM_prt_ckpt_filepath,
            infer_filepath=infer_filepath,
            train_idx=data['train_idx'],
            valid_idx=data['valid_idx'],
            test_idx=data['test_idx'],
            hard_labels=data['labels'],
            user_seq=data['user_text'],
            run=run,
            eval_patience=args.LM_eval_patience,
            activation=args.activation,
            stage="training"
        )


        LMTrainer.build_model()
        
        target_dataset_name = "dblp"
        target_data_filepath = f"./data/target_dataset/{target_dataset_name}/"
        target_dataset = open_txt_file(target_data_filepath+dataset_filepath)
        
        LMTrainer.generate_initial_embeddings(target_dataset)
        
        GNNTrainer = GNN_Trainer(
            device=args.device,
            optimizer_name=args.optimizer_GNN,
            lr=args.lr_GNN,
            weight_decay=args.weight_decay_GNN,
            dropout=args.GNN_dropout,
            pl_weight=args.beta,
            batch_size=args.batch_size_GNN,
            gnn_n_layers=args.n_layers,
            n_relations=args.n_relations,
            activation=args.activation,
            gnn_epochs_per_iter=args.GNN_epochs_per_iter,
            temperature=args.temperature,
            pl_ratio=args.pl_ratio_GNN,
            intermediate_data_filepath=GNN_intermediate_data_filepath,
            ckpt_filepath=GNN_ckpt_filepath,
            train_idx=data['train_idx'],
            valid_idx=data['valid_idx'],
            test_idx=data['test_idx'],
            hard_labels=data['labels'],
            edge_index=data['edge_index'],
            edge_type=data['edge_type'],
            run=run,
            att_heads=args.att_heads,
            gnn_hidden_dim=args.hidden_dim,
            out_channels = args.GNN_output_size
        )
        GNNTrainer.build_model()

        for iter in range(args.max_iters):
            print(f'------Iter: {iter}/{args.max_iters-1}------')
            
            embeddings_LM, soft_labels_LM = load_distilled_knowledge('LM', LM_intermediate_data_filepath, iter-1)

            # GNN training
            flag = GNNTrainer.train(embeddings_LM, soft_labels_LM, iter)
            GNNTrainer.infer(embeddings_LM)

            target_soft_labels_GNN = load_distilled_knowledge('GNN', GNN_intermediate_data_filepath, iter)
            target_length = len(target_soft_labels_GNN)
            
            remain_length = len(embeddings_LM) - target_length
            fake_labels = np.array([0 for i in range(remain_length)])
            one_hot_labels = np.eye(args.GNN_output_size)[fake_labels]
            one_hot_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)
            soft_labels_GNN = torch.cat((target_soft_labels_GNN, one_hot_tensor), dim=0)

            # LM training
            flag = LMTrainer.train(soft_labels_GNN)
            if flag:
                print(f'Early stop by LM at iter {iter}!')
                break

            # LM-GNN contrastive alignment
            gnn_ckpt = torch.load(GNNTrainer.ckpt_filepath+'best.pkl')
            GNNTrainer.model.load_state_dict(gnn_ckpt['model'])
            lm_ckpt = torch.load(LMTrainer.ckpt_filepath+'best.pkl')
            LMTrainer.model.load_state_dict(lm_ckpt['model'])

            all_idx = torch.tensor([i for i in range(target_length)])
            indices = torch.randperm(all_idx.size(0))
            shuffled_infer_idx = all_idx[indices]
            train_lm = True
            
            for t in range(25):
                if GNNTrainer.best_valid_acc < LMTrainer.best_valid_acc:
                    GNNTrainer.model.train()
                    train_lm = False
                    clip_params = [
                    {"params": GNNTrainer.model.parameters()}
                    ]
                    clip_optimizer = get_optimizer(clip_params)
                else:
                    LMTrainer.model.train()
                    train_lm = True
                    clip_params = [
                    {"params": LMTrainer.model.parameters()},
                    ]
                    clip_optimizer = get_optimizer(clip_params)
                
                epoch_loss = 0
                for bat in range(target_length//64):
                    infer_idx = shuffled_infer_idx[bat*64:(bat+1)*64]

                    gnn_outputs = GNNTrainer.clip_infer(embeddings_LM, infer_idx)
                    lm_outputs = LMTrainer.clip_infer(infer_idx)

                    gnn_final_outputs = gnn_outputs / gnn_outputs.norm(dim=-1, keepdim=True)
                    lm_final_outputs = lm_outputs / lm_outputs.norm(dim=-1, keepdim=True)

                    labels = torch.arange(gnn_final_outputs.shape[0]).to(args.device)
                    clip_loss = cal_cl_loss(gnn_final_outputs, lm_final_outputs, labels)

                    clip_optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    clip_loss.backward()
                    clip_optimizer.step()

                    epoch_loss += round((clip_loss.detach().clone()).cpu().item(), 4)
                    
                epoch_loss = epoch_loss / (target_length//64)
                print(f"loss: {epoch_loss}")

                lm_valid_acc, lm_valid_mi_f1, lm_valid_ma_f1 = LMTrainer.eval()
                print(f'after clip, LM Valid Accuracy = {lm_valid_acc}')
                print(f'after clip, LM Valid Micro F1 = {lm_valid_mi_f1}')
                print(f'after clip, LM Valid Macro F1 = {lm_valid_ma_f1}')

                gnn_valid_acc, gnn_valid_mi_f1, gnn_valid_ma_f1 = GNNTrainer.eval(embeddings_LM)
                print(f'after clip, GNN Valid Accuracy = {gnn_valid_acc}')
                print(f'after clip, GNN Valid Micro F1 = {gnn_valid_mi_f1}')
                print(f'after clip, GNN Valid Macro F1 = {gnn_valid_ma_f1}')

                if not train_lm:
                    if gnn_valid_acc > GNNTrainer.best_valid_acc:
                        GNNTrainer.best_valid_acc = gnn_valid_acc
                        print("saving GNN parameters...")
                        torch.save({'model': GNNTrainer.model.state_dict()}, GNNTrainer.ckpt_filepath+'best.pkl')
                else:
                    if lm_valid_acc > LMTrainer.best_valid_acc:
                        LMTrainer.best_valid_acc = lm_valid_acc
                        print("saving LM parameters...")
                        torch.save({'model': LMTrainer.model.state_dict()}, LMTrainer.ckpt_filepath+'best.pkl')
            
            gnn_ckpt = torch.load(GNNTrainer.ckpt_filepath+'best.pkl')
            GNNTrainer.model.load_state_dict(gnn_ckpt['model'])

            LMTrainer.infer()
        
        print(f'Best LM is iter {LMTrainer.best_iter} epoch {LMTrainer.best_epoch}!')
        LMTrainer.test()
        
        print(f'Best GNN is iter {GNNTrainer.best_iter} epoch {GNNTrainer.best_epoch}!')
        embeddings_LM = LMTrainer.load_embedding(GNNTrainer.best_iter-1)
        GNNTrainer.test(embeddings_LM)
        
        GNNTrainer.save_results(data_dir+experiment_name+f'/results_GNN.json')
        LMTrainer.save_results(data_dir+experiment_name+f'/results_LM.json')
        
if __name__ == '__main__':
    args = parser_args()
    main(args)