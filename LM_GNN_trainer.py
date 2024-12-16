import os
import json
import torch
import pickle
import numpy as np
import torch.nn.functional as F


from tqdm import tqdm
from dgl.data.utils import load_graphs
from torch.nn import CrossEntropyLoss, KLDivLoss
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_building import build_LM_model, build_GNN_model
from transformers.optimization import get_cosine_schedule_with_warmup
from dataloader import build_LM_dataloader, build_GNN_dataloader


def open_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        file_content = pickle.load(file)
        return file_content


def save_pkl_file(file_path, contents):
    with open(file_path, 'wb') as file:
        pickle.dump(contents, file)
    print("having saved pkl...")


class LM_Trainer:
    def __init__(
            self, 
            output_size,
            classifier_n_layers,
            classifier_hidden_dim,
            device, 
            pretrain_epochs,
            optimizer_name,
            lr,
            weight_decay,
            dropout,
            att_dropout,
            lm_dropout,
            activation,
            warmup,
            label_smoothing_factor,
            pl_weight,
            max_length,
            batch_size,
            grad_accumulation,
            lm_epochs_per_iter,
            temperature,
            pl_ratio,
            eval_patience,
            intermediate_data_filepath,
            ckpt_filepath,
            pretrain_ckpt_filepath,
            infer_filepath,
            train_idx,
            valid_idx,
            test_idx,
            hard_labels,
            user_seq,
            run,
            stage):
        
        self.output_size = output_size
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lm_dropout = lm_dropout
        self.warmup = warmup
        self.label_smoothing_factor = label_smoothing_factor
        self.pl_weight = pl_weight
        self.max_length = max_length
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.lm_epochs_per_iter = lm_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.eval_patience = eval_patience
        self.intermediate_data_filepath = intermediate_data_filepath    
        self.ckpt_filepath = ckpt_filepath
        self.pretrain_ckpt_filepath = pretrain_ckpt_filepath
        self.infer_filepath = infer_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.user_seq = user_seq
        self.run = run
        self.do_mlm_task = False
        
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_epoch = 0
        self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing_factor)
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')
        self.results = {}

        if stage == "fine_tuning":
            self.get_train_idx_all_for_fine_tuning()
        else:
            self.get_train_idx_all()
        self.pretrain_steps_per_epoch = self.train_idx.shape[0] // self.batch_size + 1
        self.pretrain_steps = int(self.pretrain_steps_per_epoch * self.pretrain_epochs)
        self.train_steps_per_iter = (self.train_idx_all.shape[0] // self.batch_size + 1) * self.lm_epochs_per_iter
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

        self.model_config = {
            'dropout': dropout,
            'att_dropout': att_dropout,
            'lm_dropout': self.lm_dropout,
            'classifier_n_layers': classifier_n_layers,
            'classifier_hidden_dim': classifier_hidden_dim,
            'activation': activation,
            'device': device,
            'return_mlm_loss': True if self.do_mlm_task else False,
            'output_size': self.output_size
            }
        
        self.dataloader_config = {
            'batch_size': batch_size,
            'pl_ratio': pl_ratio
            }
        
        
    def build_model(self):
        self.model, self.tokenizer = build_LM_model(self.model_config)
        self.SEP_id = self.tokenizer.convert_tokens_to_ids('SEP:')
       
    def get_optimizer(self, parameters):
        
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(parameters, **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(parameters, **self.optimizer_args)
        else:
            return NotImplementedError
        
        return optimizer
    
    def get_scheduler(self, optimizer, mode='train'):
        if mode == 'pretrain':
            return get_cosine_schedule_with_warmup(optimizer, self.pretrain_steps_per_epoch * self.warmup, self.pretrain_steps) 
        else:
            return CosineAnnealingLR(optimizer, T_max=self.train_steps_per_iter, eta_min=0)
        
    
    def pretrain(self):
        print('LM pretraining start!')
        optimizer = self.get_optimizer(self.model.parameters())
        scheduler = self.get_scheduler(optimizer, 'pretrain')

        step = 0
        valid_acc_best = 0
        valid_step_best = 0
        
        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.pretrain_ckpt_filepath+'best.pkl')
        
        train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx, self.user_seq, self.hard_labels, 'pretrain')

        for epoch in range(int(self.pretrain_epochs)+1):
            self.model.train()
            print(f'------LM Pretraining Epoch: {epoch}/{int(self.pretrain_epochs)}------')
            for batch in tqdm(train_loader):
                step += 1
                if step >= self.pretrain_steps:
                    break
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                _, output = self.model(tokenized_tensors)
                loss = self.criterion(output, labels)
                loss /= self.grad_accumulation
                loss.backward()
                self.run.log({'LM Pretrain Loss': loss.item()})
                
                if step % self.grad_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()

                if step % self.eval_patience == 0:
                    valid_acc, valid_mi_f1, valid_ma_f1 = self.eval()

                    print(f'LM Pretrain Valid Accuracy = {valid_acc}')
                    print(f'LM Pretrain Valid Micro F1 = {valid_mi_f1}')
                    print(f'LM Pretrain Valid Macro F1 = {valid_ma_f1}')
                    self.run.log({'LM Pretrain Valid Accuracy': valid_acc})
                    self.run.log({'LM Pretrain Valid Micro F1': valid_mi_f1})
                    self.run.log({'LM Pretrain Valid Macro F1': valid_ma_f1})

                    if valid_acc > valid_acc_best:
                        valid_acc_best = valid_acc
                        valid_step_best = step
                        
                        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.pretrain_ckpt_filepath+'best.pkl')
                
        
        print(f'The highest pretrain valid accuracy is {valid_acc_best}!')
        print(f'Load model from step {valid_step_best}')
        self.model.eval()
        all_outputs = []
        all_labels = []
        embeddings = []
        infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.hard_labels, mode='infer')
        with torch.no_grad():
            ckpt = torch.load(self.pretrain_ckpt_filepath+'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            for batch in tqdm(infer_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                embedding, output = self.model(tokenized_tensors)
                embeddings.append(embedding.detach().cpu())
                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            embeddings = torch.cat(embeddings, dim=0)
            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]

            test_predictions = torch.argmax(all_outputs[self.test_idx], dim=1).numpy()
            test_labels = torch.argmax(all_labels[self.test_idx], dim=1).numpy()
            torch.save(embeddings, self.intermediate_data_filepath+'embeddings_iter_-1.pt')
            torch.save(soft_labels, self.intermediate_data_filepath+'soft_labels_iter_-1.pt')

            test_acc = accuracy_score(test_predictions, test_labels)
            test_mi_f1 = f1_score(test_predictions, test_labels, average="micro")
            test_ma_f1 = f1_score(test_predictions, test_labels, average="macro")
            self.results['pretrain accuracy'] = test_acc
            self.results['pretrain maf1'] = test_ma_f1
        

        print(f'LM Pretrain Test Accuracy = {test_acc}')
        print(f'LM Pretrain Test Micro F1 = {test_mi_f1}')
        print(f'LM Pretrain Test Macro F1 = {test_ma_f1}')
        self.run.log({'LM Pretrain Test Accuracy': test_acc})
        self.run.log({'LM Pretrain Test Micro F1': test_mi_f1})
        self.run.log({'LM Pretrain Test Macro F1': test_ma_f1})
    

    def generate_initial_embeddings(self, target_dataset):
        self.model.eval()
        all_labels = []
        all_outputs = []
        embeddings = []
        
        infer_loader = build_LM_dataloader(self.dataloader_config, None, target_dataset, self.hard_labels, mode='infer')
        
        with torch.no_grad():
            ckpt = torch.load(self.pretrain_ckpt_filepath+'best.pkl')
            
            new_state_dict = self.model.state_dict()
            for name, param in ckpt['model'].items():
                if 'classifier' not in name:
                    new_state_dict[name] = param
            self.model.load_state_dict(new_state_dict)
            
            for batch in tqdm(infer_loader):  # no shuffle
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                embedding, output = self.model(tokenized_tensors)
                embeddings.append(embedding.detach().cpu())
                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            embeddings = torch.cat(embeddings, dim=0)
            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]
            
            torch.save(embeddings, self.intermediate_data_filepath+'embeddings_iter_-1.pt')
            torch.save(soft_labels, self.intermediate_data_filepath+'soft_labels_iter_-1.pt')
            print("having saved initial embeddings of the target dataset...")


    def train(self, soft_labels):
        optimizer = self.get_optimizer(self.model.parameters())
        scheduler = self.get_scheduler(optimizer)
        
        early_stop_flag = True
        print('LM training start!')
        step = 0
        train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx_all, self.user_seq, soft_labels, 'train', self.is_pl)

        for epoch in range(self.lm_epochs_per_iter):

            self.model.train()
            
            print(f'This is iter {self.iter} epoch {epoch}/{self.lm_epochs_per_iter-1}')

            for batch in tqdm(train_loader):
                step += 1
                
                tokenized_tensors, labels, is_pl = self.batch_to_tensor(batch)

                lm_embs, output = self.model(tokenized_tensors)
                
                pl_idx = torch.nonzero(is_pl == 1).squeeze()  # soft label
                rl_idx = torch.nonzero(is_pl == 0).squeeze()  # hard label

                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], labels[rl_idx])
                elif rl_idx.numel() == 0:  
                    temp = F.log_softmax(output[pl_idx] / self.temperature, dim=-1)
                    loss = self.KD_criterion(temp, labels[pl_idx])
                else:
                    temp = F.log_softmax(output[pl_idx] / self.temperature, dim=-1)
                    loss_KD = self.KD_criterion(temp, labels[pl_idx])
                    loss_H = self.criterion(output[rl_idx], labels[rl_idx])
                    self.run.log({'loss_KD': loss_KD.item()})
                    self.run.log({'loss_H': loss_H.item()})
                    loss = self.pl_weight * loss_KD + (1 - self.pl_weight) * loss_H
                
                loss /= self.grad_accumulation
                loss.backward(retain_graph=True)
                self.run.log({'LM Train Loss': loss.item()})
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
                valid_acc, valid_mi_f1, valid_ma_f1 = self.eval()

                print(f'LM Valid Accuracy = {valid_acc}')
                print(f'LM Valid Micro F1 = {valid_mi_f1}')
                print(f'LM Valid Macro F1 = {valid_ma_f1}')
                self.run.log({'LM Valid Accuracy': valid_acc})
                self.run.log({'LM Valid F1': valid_mi_f1})
                self.run.log({'LM Valid F1': valid_ma_f1})

                if valid_acc > self.best_valid_acc:
                    early_stop_flag = False
                    self.best_valid_acc = valid_acc
                    self.best_iter = self.iter
                    self.best_epoch = epoch
                    print("saving LM parameters...")
                    torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.ckpt_filepath+'best.pkl')
                
        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        return early_stop_flag
    
    def infer(self):
        self.model.eval()

        infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.hard_labels, mode='infer')
        all_outputs = []
        all_labels = []
        embeddings = []
        with torch.no_grad():
            ckpt = torch.load(self.ckpt_filepath+'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            for batch in tqdm(infer_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                embedding, output = self.model(tokenized_tensors)
                embeddings.append(embedding.detach().cpu())
                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            embeddings = torch.cat(embeddings, dim=0)
    
            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]

            torch.save(soft_labels, self.intermediate_data_filepath+f'soft_labels_iter_{self.iter}.pt')
            torch.save(embeddings, self.intermediate_data_filepath+f'embeddings_iter_{self.iter}.pt')

            self.iter += 1

    
    def clip_infer(self, infer_idx):
        
        infer_loader = build_LM_dataloader(self.dataloader_config, infer_idx, self.user_seq, self.hard_labels, mode='clip_infer')
        
        for batch in tqdm(infer_loader):
            tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

            output, _ = self.model(tokenized_tensors)  # emb
            # _, output = self.model(tokenized_tensors)
            return output


    def eval(self, mode='valid'):
        if mode == 'valid':            
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.valid_idx, self.user_seq, self.hard_labels, mode='eval')
        elif mode == 'test':
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.test_idx, self.user_seq, self.hard_labels, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                _, output = self.model(tokenized_tensors)

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_mi_f1 = f1_score(valid_labels, valid_predictions, average='micro')
            valid_ma_f1 = f1_score(valid_labels, valid_predictions, average='macro')

            return valid_acc, valid_mi_f1, valid_ma_f1


    def test(self):
        print('Computing test accuracy and f1 for LM...')
        ckpt = torch.load(self.ckpt_filepath+'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_mi_f1, test_ma_f1 = self.eval('test')
        print(f'LM Test Accuracy = {test_acc}')
        print(f'LM Test Micro F1 = {test_mi_f1}')
        print(f'LM Test Macro F1 = {test_ma_f1}')
        self.run.log({'LM Test Accuracy': test_acc})
        self.run.log({'LM Test Micro F1': test_mi_f1})
        self.run.log({'LM Test Macro F1': test_ma_f1})
        self.results['accuracy'] = test_acc
        self.results['mif1'] = test_mi_f1
        self.results['maf1'] = test_ma_f1


    def batch_to_tensor(self, batch):
                    
        tokenized_tensors = self.tokenizer(text=batch[0], return_tensors='pt', max_length=self.max_length, truncation=True, padding='longest', add_special_tokens=False)
        for key in tokenized_tensors.keys():
            tokenized_tensors[key] = tokenized_tensors[key].to(self.device)
        labels = batch[1].to(self.device)
    
        if len(batch) == 3:
            is_pl = batch[2].to(self.device)
            return tokenized_tensors, labels, is_pl
        else:
            return tokenized_tensors, labels, None
    

    def class_tensor(self, class_texts):
                    
        tokenized_tensors = self.tokenizer(text=class_texts, return_tensors='pt', max_length=self.max_length, truncation=True, padding='longest', add_special_tokens=False)
        for key in tokenized_tensors.keys():
            tokenized_tensors[key] = tokenized_tensors[key].to(self.device)
        return tokenized_tensors
        

    def load_embedding(self, iter):
        embeddings = torch.load(self.intermediate_data_filepath+f'embeddings_iter_{iter}.pt')
        return embeddings
    

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)
        

    def get_train_idx_all(self, target_type_num=4057):
        glist, label_dict = load_graphs('./data/data_for_fine_tuning/graph.bin')
        g = glist[0]
        label = g.nodes['author'].data['label'].tolist()
        labeled_node_ids = {0:[], 1:[], 2:[], 3:[]}
        num = len(label)
        for i in range(num):
            labeled_node_ids[label[i]].append(i)
        all_labeled_idx = []
        for i in [0, 1, 2]:
            all_labeled_idx.extend(labeled_node_ids[i])

        n_total = len(all_labeled_idx)
        all = set(all_labeled_idx)
        exclude = set(self.train_idx.numpy())
        n = self.train_idx.shape[0]
        pl_ratio_LM = min(self.pl_ratio, (n_total - n) / n)
        n_pl_LM = int(n_total * pl_ratio_LM)
        pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_LM, replace=False))
        self.train_idx_all = torch.cat((self.train_idx, pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0


    def get_train_idx_all_for_fine_tuning(self):
        n_total = self.hard_labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.numpy())
        n = self.train_idx.shape[0]
        pl_ratio_LM = min(self.pl_ratio, (n_total - n) / n)
        n_pl_LM = int(n * pl_ratio_LM)
        pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_LM, replace=False))
        self.train_idx_all = torch.cat((self.train_idx, pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0


class GNN_Trainer:
    def __init__(
        self, 
        device, 
        optimizer_name,
        lr,
        weight_decay,
        dropout,
        pl_weight,
        batch_size,
        gnn_n_layers,
        n_relations,
        activation,
        gnn_epochs_per_iter,
        temperature,
        pl_ratio,
        intermediate_data_filepath,
        ckpt_filepath,
        train_idx,
        valid_idx,
        test_idx,
        hard_labels,
        edge_index, 
        edge_type,
        run,
        att_heads,
        gnn_hidden_dim,
        out_channels
        ):
    
        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.pl_weight = pl_weight
        self.dropout = dropout        
        self.batch_size = batch_size
        self.gnn_n_layers = gnn_n_layers
        self.n_relations = n_relations
        self.activation = activation
        self.gnn_epochs_per_iter = gnn_epochs_per_iter
        self.temperature = temperature
        self.pl_ratio = pl_ratio
        self.intermediate_data_filepath = intermediate_data_filepath    
        self.ckpt_filepath = ckpt_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.run = run
        self.att_heads = att_heads
        self.gnn_hidden_dim = gnn_hidden_dim
        self.lm_input_dim = 768
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.KD_criterion = KLDivLoss(log_target=False, reduction='batchmean')

        
        self.results = {}
        self.get_train_idx_all()
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)
        
        self.model_config = {
            'optimizer': optimizer_name,
            'gnn_n_layers': gnn_n_layers,
            'n_relations': n_relations,
            'activation': activation,
            'dropout': dropout,
            'gnn_hidden_dim': gnn_hidden_dim,
            'lm_input_dim': self.lm_input_dim,
            'att_heads': att_heads,
            'device': device,
            'out_channels': out_channels
            }
        
        self.dataloader_config = {
            'batch_size': batch_size,
            'n_layers': gnn_n_layers
            }
        
    

    def build_model(self):
        self.model = build_GNN_model(self.model_config)

    def get_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.gnn_epochs_per_iter, eta_min=0)
    

    def get_optimizer(self):
        
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), **self.optimizer_args)
        else:
            return NotImplementedError
        
        return optimizer
   
    def train(self, embeddings_LM, soft_labels, iter):

        early_stop_flag = True
        
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        print('GNN training start!')
        print(f'This is iter {self.iter}')

        train_loader = build_GNN_dataloader(self.dataloader_config, self.train_idx_all, embeddings_LM, soft_labels, self.edge_index, self.edge_type, mode='train', is_pl=self.is_pl)
        
        for epoch in tqdm(range(self.gnn_epochs_per_iter)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                is_pl = batch.is_pl[0: batch_size].to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)

                _, output = self.model(x_batch, edge_index_batch, edge_type_batch) 
                output = output[0: batch_size]

                pl_idx = torch.nonzero(is_pl == 1).squeeze()
                rl_idx = torch.nonzero(is_pl == 0).squeeze()

        
                if pl_idx.numel() == 0:
                    loss = self.criterion(output[rl_idx], labels[rl_idx])
                elif rl_idx.numel() == 0:
                    loss = self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx])
                else:
                    loss = self.pl_weight * self.KD_criterion(F.log_softmax(output[pl_idx] / self.temperature, dim=-1), labels[pl_idx]) + (1 - self.pl_weight) * self.criterion(output[rl_idx], labels[rl_idx])

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'GNN Train Loss': loss.item()})
            

            valid_acc, valid_mi_f1, valid_ma_f1 = self.eval(embeddings_LM)
     
            self.run.log({'GNN Valid Accuracy': valid_acc})
            self.run.log({'GNN Valid Micro F1': valid_mi_f1})
            self.run.log({'GNN Valid Macro F1': valid_ma_f1})

            if valid_acc > self.best_valid_acc:
                early_stop_flag = False
                self.best_valid_acc = valid_acc
                self.best_epoch = epoch
                self.best_iter = self.iter
                print("saving GNN parameters...")
                torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.ckpt_filepath+'best.pkl')
        print(f'The highest valid accuracy is {self.best_valid_acc}!')
        return early_stop_flag
    
    def infer(self, embeddings_LM):
        self.model.eval()
        infer_idx = torch.tensor([i for i in range(4057)])
        infer_loader = build_GNN_dataloader(self.dataloader_config, infer_idx, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='infer')

        all_outputs = []
        all_labels = []
        with torch.no_grad():
            ckpt = torch.load(self.ckpt_filepath+'best.pkl')
            self.model.load_state_dict(ckpt['model'])
            for batch in infer_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)
                
                _, output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            soft_labels = torch.softmax(all_outputs / self.temperature, dim=1)
            soft_labels[self.train_idx] = all_labels[self.train_idx]

            torch.save(soft_labels, self.intermediate_data_filepath+f'soft_labels_iter_{self.iter}.pt')

            self.iter += 1


    def clip_infer(self, embeddings_LM, infer_idx):
        infer_loader = build_GNN_dataloader(self.dataloader_config, infer_idx, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='infer')

        for batch in infer_loader:
            batch_size = batch.batch_size
            x_batch = batch.x.to(self.device)

            edge_index_batch = batch.edge_index.to(self.device)
            edge_type_batch = batch.edge_type.to(self.device)
            
            output, _ = self.model(x_batch, edge_index_batch, edge_type_batch)
            # _, output = self.model(x_batch, edge_index_batch, edge_type_batch)
            output = output[0: batch_size]

            return output


    def eval(self, embeddings_LM, mode='valid'):
        if mode == 'valid':            
            eval_loader = build_GNN_dataloader(self.dataloader_config, self.valid_idx, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='eval')
        elif mode == 'test':
            eval_loader = build_GNN_dataloader(self.dataloader_config, self.test_idx, embeddings_LM, self.hard_labels, self.edge_index, self.edge_type, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)
                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)
                
                _, output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_mi_f1 = f1_score(valid_labels, valid_predictions, average='micro')
            valid_ma_f1 = f1_score(valid_labels, valid_predictions, average='macro')

            return valid_acc, valid_mi_f1, valid_ma_f1
        

    def test(self, embeddings_LM):
        print('Computing test accuracy and f1 for GNN...')
        ckpt = torch.load(self.ckpt_filepath+'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_mi_f1, test_ma_f1 = self.eval(embeddings_LM, 'test')
        print(f'GNN Test Accuracy = {test_acc}')
        print(f'GNN Test Micro F1 = {test_mi_f1}')
        print(f'GNN Test Macro F1 = {test_ma_f1}')
        self.run.log({'GNN Test Accuracy': test_acc})
        self.run.log({'GNN Test Micro F1': test_mi_f1})
        self.run.log({'GNN Test Macro F1': test_ma_f1})
        self.results['accuracy'] = test_acc
        self.results['mif1'] = test_mi_f1
        self.results['maf1'] = test_ma_f1


    def load_soft_labels(self, iter):
        soft_labels = torch.load(self.intermediate_data_filepath+f'soft_labels_iter_{iter}.pt')
        return soft_labels

    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)
        
    def get_train_idx_all(self):
        n_total = self.hard_labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.numpy())
        n = self.train_idx.shape[0]
        pl_ratio_GNN = min(self.pl_ratio, (n_total - n) / n)
        n_pl_GNN = int(n * pl_ratio_GNN)
        self.pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_GNN, replace=False))
        
        self.train_idx_all = torch.cat((self.train_idx, self.pl_idx))
        self.is_pl = torch.ones_like(self.train_idx_all, dtype=torch.int64)
        self.is_pl[0: n] = 0