import torch
import pickle
import random
import numpy as np
from dgl.data.utils import load_graphs


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
 

def construct_labels(classes=['0', '1', '2']):

    labels = open_txt_file("../data/data_for_fine_tuning/labels.txt")
    for i in range(len(labels)):
        if labels[i] not in classes:  # random pick N classes
            labels[i] = '0'
    
    # 补充剩余的内容
    glist, label_dict = load_graphs('../data/data_for_fine_tuning/graph.bin')
    g = glist[0]
    remain = g.num_nodes('conf') + g.num_nodes('paper') + g.num_nodes('term')
    for i in range(remain):
        labels.append("0")
    
    labels = np.array([int(i) for i in labels])

    one_hot_labels = np.eye(np.max(labels) + 1)[labels]
    one_hot_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)
    # save_pkl_file("../data/target_dataset/dblp/labels.pkl", one_hot_tensor)


def construct_edges():
    # paper2author 0
    # paper2term 1
    # paper2conf 2
    # author2paper 3
    # term2paper 4
    # conf2paper 5
    src, dst = [], []
    edge_type = []
    with open('../data/data_for_fine_tuning/link.dat', 'r') as original_meta_file:
        for line in original_meta_file:
            start, end, et = line.split('\t')
            src.append(int(start))
            dst.append(int(end))
            edge_type.append(int(et))
    # save_pkl_file("../data/target_dataset/dblp/edge_index.pkl", torch.tensor([src, dst]))
    # save_pkl_file("../data/target_dataset/dblp/edge_type.pkl", torch.tensor(edge_type))


def all_pick_ids(node_ids):
    all_ids = []
    for ids in node_ids:
        all_ids.extend(ids)
    return all_ids


def construct_few_shot_idx():
    glist, label_dict = load_graphs('../data/data_for_fine_tuning/graph.bin')
    g = glist[0]
    label = g.nodes['author'].data['label'].tolist()
    labeled_node_ids = {0:[], 1:[], 2:[], 3:[]}
    num = len(label)
    for i in range(num):
        labeled_node_ids[label[i]].append(i)

    n_way = 3
    k_shot = 3
    train_node_ids = []
    val_node_ids = []
    test_node_ids = []

    for i in range(n_way):
        train_pick_ids = random.sample(labeled_node_ids[i], k_shot)
        train_node_ids.append(train_pick_ids)
        remainning_ids = list(set(labeled_node_ids[i])-set(train_pick_ids))
        val_pick_ids = random.sample(remainning_ids, int(len(remainning_ids)/2))
        val_node_ids.append(val_pick_ids)
        test_pick_ids = list(set(remainning_ids)-set(val_pick_ids))
        test_node_ids.append(test_pick_ids)

    all_train_ids = all_pick_ids(train_node_ids)
    all_val_ids = all_pick_ids(val_node_ids)
    all_test_ids = all_pick_ids(test_node_ids)
    # save_pkl_file("../data/target_dataset/dblp/train_idx.pkl", torch.tensor(all_train_ids))
    # save_pkl_file("../data/target_dataset/dblp/valid_idx.pkl", torch.tensor(all_val_ids))
    # save_pkl_file("../data/target_dataset/dblp/test_idx.pkl", torch.tensor(all_test_ids))


if __name__ == "__main__":
    construct_edges()
    construct_few_shot_idx()
    construct_labels()