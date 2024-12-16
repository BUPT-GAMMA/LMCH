import dgl
import pickle
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

def extract_all_node_metapath(data_dir, target_type, metapaths, relation, mid_types):
    # load graph
    glist, label_dict = load_graphs(data_dir + 'graph.bin')
    g = glist[0]

    graph_node_name = open_pkl_file(data_dir + 'graph_node_name.pkl')
    metapath = metapaths[target_type]
    relation = relation[target_type]
    
    sampling_time = 10
    num_nodes = g.num_nodes(target_type)
        
    all_path_for_sampling_times = [[] for _ in range(num_nodes)]
    print("---------------------------------------")
    print(f"Sampling nodes of type {target_type}...")
    for p, path in enumerate(metapath):
        path_for_sampling_times = [[] for _ in range(num_nodes)]
        print(f"Sampling the {p}th path...")
        for st in range(sampling_time):
            traces, types = dgl.sampling.random_walk(g=g, nodes=g.nodes(target_type), metapath=path)
            traces = traces.tolist()
            length = len(traces)
            print(f"Performing the {st}th sampling...")
            for node in range(length):
                if node % 5000 == 0:
                    print(f"{node} nodes have been sampled...")
                path_i = []
                if traces[node][1] != -1:
                    path_i.append(graph_node_name[target_type][traces[node][0]].replace('_', ' '))
                    path_i.append(relation[p][0])
                    mid_type = mid_types[target_type][p][0]
                    path_i.append(graph_node_name[mid_type][traces[node][1]].replace('_', ' '))
                    path_i.append(relation[p][1])
                    if len(traces[node]) <= 3:  # apa pa
                        path_i.append(graph_node_name[target_type][traces[node][2]].replace('_', ' '))
                    else:  # apcpa pcpa
                        mid_type = mid_types[target_type][p][1]
                        path_i.append(graph_node_name[mid_type][traces[node][2]].replace('_', ' '))
                        path_i.append(relation[p][2])

                        mid_type = mid_types[target_type][p][2]
                        path_i.append(graph_node_name[mid_type][traces[node][3]].replace('_', ' '))
                        path_i.append(relation[p][3])

                        mid_type = target_type
                        path_i.append(graph_node_name[mid_type][traces[node][4]].replace('_', ' '))
                else:
                    path_i.append(graph_node_name[target_type][traces[node][0]].replace('_', ' '))
                
                path_i.append("[SEP]")
                path_i = " ".join(path_i)
                path_for_sampling_times[node].append(path_i)
        path_for_sampling_times = [list(set(item)) for item in path_for_sampling_times if item]
        for i, item in enumerate(path_for_sampling_times):
            path_for_sampling_times[i] = " ".join(item)
            all_path_for_sampling_times[i].append(path_for_sampling_times[i])
    all_path_for_sampling_times = [list(set(item)) for item in all_path_for_sampling_times if item]
    for i, item in enumerate(all_path_for_sampling_times):
        all_path_for_sampling_times[i] = " ".join(item)
        all_path_for_sampling_times[i].rstrip(' [SEP] ')
    
    print(all_path_for_sampling_times[0])
    print(f"length: {len(all_path_for_sampling_times)}")
    return all_path_for_sampling_times


if __name__ == "__main__":
    # dblp 4057
    # ('paper', 'was written by', 'author')
    # ('paper', 'was published in', 'term')
    # ('paper', 'was received by', 'conf')
    # ('author', 'write', 'paper')
    # ('term', 'publish', 'paper')
    # ('conf', 'receive', 'paper')
    data_dir = '../data/data_for_fine_tuning/'
    metapaths = {'author': [['write', 'was written by'], 
                            ['write', 'was received by', 'receive', 'was written by'],
                            ['write', 'was published in', 'publish', 'was written by']],  # apa apcpa aptpa
                 'conf': [['receive', 'was received by'], 
                          ['receive', 'was written by', 'write', 'was received by'],
                          ['receive', 'was published in', 'publish', 'was received by']],  # cpc cpapc cptpc
                 'paper': [['was written by', 'write'],
                           ['was published in', 'publish'],
                           ['was received by', 'receive']],  # pap ptp pcp
                 'term':[['publish', 'was published in'], 
                         ['publish', 'was received by', 'receive', 'was published in'],
                         ['publish', 'was written by', 'write', 'was published in']]}  # tpt tpcpt tpapt
    relation = {'author': [['write', 'was written by'], 
                            ['write', 'was received by', 'receive', 'was written by'],
                            ['write', 'was published in', 'publish', 'was written by']],  # apa apcpa aptpa
                 'conf': [['receive', 'was received by'], 
                          ['receive', 'was written by', 'write', 'was received by'],
                          ['receive', 'was published in', 'publish', 'was received by']],  # cpc cpapc cptpc
                 'paper': [['was written by', 'write'],
                           ['was published in', 'publish'],
                           ['was received by', 'receive']],  # pap ptp pcp
                 'term': [['publish', 'was published in'], 
                          ['publish', 'was received by', 'receive', 'was published in'],
                          ['publish', 'was written by', 'write', 'was published in']]}  # tpt tpcpt tpapt
    mid_types = {'author': [['paper'], ['paper', 'conf', 'paper'], ['paper', 'term', 'paper']],
                 'conf': [['paper'], ['paper', 'author', 'paper'], ['paper', 'term', 'paper']],
                 'paper': [['author'], ['term'], ['conf']],
                 'term': [['paper'], ['paper', 'conf', 'paper'], ['paper', 'author', 'paper']]}

    all_corpus = []
    author_corpus = extract_all_node_metapath(data_dir, 'author', metapaths, relation, mid_types)
    conf_corpus = extract_all_node_metapath(data_dir, 'conf', metapaths, relation, mid_types)
    paper_corpus = extract_all_node_metapath(data_dir, 'paper', metapaths, relation, mid_types)
    term_corpus = extract_all_node_metapath(data_dir, 'term', metapaths, relation, mid_types)
    all_corpus.extend(author_corpus)
    all_corpus.extend(conf_corpus)
    all_corpus.extend(paper_corpus)
    all_corpus.extend(term_corpus)
    # save_txt_file(data_dir + 'metapath_corpus.txt', all_corpus)
    print(all_corpus[0])
    print(f"length: {len(all_corpus)}")  # 26128

    