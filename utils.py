import torch
import wandb
import pickle
import random, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def setup_wandb(args, experiment_name, seed):
    run = wandb.init(
        project="LMCH",
        name=experiment_name + f'_seed_{seed}',
        config=args,
        mode="disabled"
    )
    return run


def open_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        file_content = pickle.load(file)
        return file_content


def save_pkl_file(file_path, contents):
    with open(file_path, 'wb') as file:
        pickle.dump(contents, file)
    print("having saved pkl...")


def open_txt_file(file_path):
    with open(file_path, 'r') as file:
        contents = [line.rstrip("\n") for line in file.readlines()]
        return contents


def save_txt_file(file_path, contents):
    with open(file_path, 'w') as file:
        for paragraph in contents:
            file.write(paragraph + "\n")
    print("having saved txt...")
    

def load_raw_data(dataset_filepath, label_filepath):
    
    data_filepath = "./data/data_for_fine_tuning/"
    print('Loading data...')
    train_idx = open_pkl_file(data_filepath+'train_idx.pkl')
    valid_idx = open_pkl_file(data_filepath+'valid_idx.pkl')
    test_idx = open_pkl_file(data_filepath+'test_idx.pkl')

    user_text = open_txt_file(data_filepath+dataset_filepath)
    labels = torch.tensor(open_pkl_file(data_filepath+label_filepath))

    return {'train_idx': train_idx, 
            'valid_idx': valid_idx, 
            'test_idx': test_idx, 
            'user_text': user_text, 
            'labels': labels}


def load_few_shot_data(target_dataset, dataset_filepath, label_filepath, use_GNN):

    data_filepath = f"./data/target_dataset/{target_dataset}/"
    print('Loading data...')

    # 3-way 3-shot
    train_idx = open_pkl_file(data_filepath+'train_idx.pkl')
    valid_idx = open_pkl_file(data_filepath+'valid_idx.pkl')
    test_idx = open_pkl_file(data_filepath+'test_idx.pkl')

    user_text = open_txt_file(data_filepath+dataset_filepath)
    labels = open_pkl_file(data_filepath+label_filepath)
    
    if use_GNN:
        edge_index = open_pkl_file(data_filepath+'edge_index.pkl')
        edge_type = open_pkl_file(data_filepath+'edge_type.pkl')
        return {'train_idx': train_idx, 
                'valid_idx': valid_idx, 
                'test_idx': test_idx, 
                'user_text': user_text, 
                'labels': labels, 
                'edge_index': edge_index,
                'edge_type': edge_type}
    else:
        return {'train_idx': train_idx, 
                'valid_idx': valid_idx, 
                'test_idx': test_idx, 
                'user_text': user_text, 
                'labels': labels}
    

def load_distilled_knowledge(from_which_model, intermediate_data_filepath, iter):
    if from_which_model == 'LM':
        embeddings = torch.load(intermediate_data_filepath+f'embeddings_iter_{iter}.pt')
        soft_labels = torch.load(intermediate_data_filepath+f'soft_labels_iter_{iter}.pt')
        return embeddings, soft_labels
    
    elif from_which_model == 'GNN':
       
        soft_labels = torch.load(intermediate_data_filepath+f'soft_labels_iter_{iter}.pt')
        return soft_labels

    elif from_which_model == 'MLP':
        soft_labels = torch.load(intermediate_data_filepath+f'soft_labels_iter_{iter}.pt')
        return soft_labels
    
    else:
        raise ValueError('"from_which_model" should be "LM", "GNN" or "MLP".')


# optimizer for clip
def get_optimizer(parameters, name="adam"):

    optimizer_args = dict(lr=2e-5)
        
    if name == "adam":
        optimizer = torch.optim.Adam(parameters, **optimizer_args)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(parameters, **optimizer_args)
    elif name == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **optimizer_args)
    elif name == "radam":
        optimizer = torch.optim.RAdam(parameters, **optimizer_args)
    else:
        return NotImplementedError
    
    return optimizer


# loss for clip
def cal_cl_loss(gnn_outputs, lm_outputs, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * gnn_outputs @ lm_outputs.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss