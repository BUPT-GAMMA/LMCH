from utils import *
from parser_args import parser_args
from LM_GNN_trainer import LM_Trainer


def main(args):
    
    for seed in list(map(int, args.seeds.strip().split(','))):

        experiment_name = f"three_wo_dblp_metapath_seed_{seed}"
        data_dir = "./"

        seed_setting(seed)
        
        infer_filepath = data_dir + "infer/"
        LM_prt_ckpt_filepath = data_dir + experiment_name + "/checkpoints/LM_pretrain/"
        LM_ckpt_filepath = data_dir + experiment_name + "/checkpoints/LM/"
        LM_intermediate_data_filepath = data_dir + experiment_name + "/intermediate/LM/"

        # pretrain for dblp
        dataset_filepath = "metapath_corpora.txt"
        label_filepath = "labels.pkl"
        data = load_raw_data(dataset_filepath, label_filepath)

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
            stage="fine_tuning"
        )

        LMTrainer.build_model()
        LMTrainer.pretrain()
        

if __name__ == '__main__':
    args = parser_args()
    main(args)