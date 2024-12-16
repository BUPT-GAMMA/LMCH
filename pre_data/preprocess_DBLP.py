import dgl
import pickle
import torch as th
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from dgl.data.utils import save_graphs
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as sklearn_stopwords

def get_src_dst(temp, src_sub, dst_sub):
    src, dst = [], []
    for i in range(len(temp)):
        src.append(temp[i][0]-src_sub)
        dst.append(temp[i][1]-dst_sub)
    return src, dst


def save_pkl_file(file_path, contents):
    with open(file_path, 'wb') as file:
        pickle.dump(contents, file)
    print("having saved pkl...")


def save_txt_file(file_path, contents):
    with open(file_path, 'w') as file:
        for paragraph in contents:
            file.write(paragraph + "\n")
    print("having saved txt...")


if __name__ == "__main__":
    save_prefix = '../data/data_for_fine_tuning/'

    author_label = pd.read_csv('../data/raw/DBLP/author_label.txt', sep='\t', header=None, names=['author_id', 'label', 'author_name'], keep_default_na=False, encoding='utf-8')
    paper_author = pd.read_csv('../data/raw/DBLP/paper_author.txt', sep='\t', header=None, names=['paper_id', 'author_id'], keep_default_na=False, encoding='utf-8')
    paper_conf = pd.read_csv('../data/raw/DBLP/paper_conf.txt', sep='\t', header=None, names=['paper_id', 'conf_id'], keep_default_na=False, encoding='utf-8')
    paper_term = pd.read_csv('../data/raw/DBLP/paper_term.txt', sep='\t', header=None, names=['paper_id', 'term_id'], keep_default_na=False, encoding='utf-8')
    papers = pd.read_csv('../data/raw/DBLP/paper.txt', sep='\t', header=None, names=['paper_id', 'paper_title'], keep_default_na=False, encoding='cp1252')
    terms = pd.read_csv('../data/raw/DBLP/term.txt', sep='\t', header=None, names=['term_id', 'term'], keep_default_na=False, encoding='utf-8')
    confs = pd.read_csv('../data/raw/DBLP/conf.txt', sep='\t', header=None, names=['conf_id', 'conf'], keep_default_na=False, encoding='utf-8')

    # filter out all nodes which does not associated with labeled authors
    labeled_authors = author_label['author_id'].to_list()
    paper_author = paper_author[paper_author['author_id'].isin(labeled_authors)].reset_index(drop=True)
    valid_papers = paper_author['paper_id'].unique()
    papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
    paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
    paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
    valid_terms = paper_term['term_id'].unique()
    terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)

    # term lemmatization and grouping
    lemmatizer = WordNetLemmatizer()
    lemma_id_mapping = {}
    lemma_list = []
    lemma_id_list = []
    i = 0
    for _, row in terms.iterrows():
        i += 1
        lemma = lemmatizer.lemmatize(row['term'])
        lemma_list.append(lemma)
        if lemma not in lemma_id_mapping:
            lemma_id_mapping[lemma] = row['term_id']
        lemma_id_list.append(lemma_id_mapping[lemma])
    terms['lemma'] = lemma_list
    terms['lemma_id'] = lemma_id_list

    term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}
    lemma_id_list = []
    for _, row in paper_term.iterrows():
        lemma_id_list.append(term_lemma_mapping[row['term_id']])
    paper_term['lemma_id'] = lemma_id_list

    paper_term = paper_term[['paper_id', 'lemma_id']]
    paper_term.columns = ['paper_id', 'term_id']
    paper_term = paper_term.drop_duplicates()
    terms = terms[['lemma_id', 'lemma']]
    terms.columns = ['term_id', 'term']
    terms = terms.drop_duplicates()

    # filter out stopwords from terms
    stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
    stopword_id_list = terms[terms['term'].isin(stopwords)]['term_id'].to_list()
    paper_term = paper_term[~(paper_term['term_id'].isin(stopword_id_list))].reset_index(drop=True)
    terms = terms[~(terms['term'].isin(stopwords))].reset_index(drop=True)

    author_label = author_label.sort_values('author_id').reset_index(drop=True)
    papers = papers.sort_values('paper_id').reset_index(drop=True)
    terms = terms.sort_values('term_id').reset_index(drop=True)
    confs = confs.sort_values('conf_id').reset_index(drop=True)

    # extract labels of authors
    labels = author_label['label'].to_numpy()

    num_author = len(author_label)
    num_conf = len(confs)
    num_paper = len(papers)
    num_term = len(terms)

    # graph construction
    graph_node_name = {'author': [], 'conf': [], 'paper': [], 'term': []}
    author_id2num = {}
    conf_id2num = {}
    paper_id2num = {}
    term_id2num = {}
    for idx, row in author_label.iterrows():
        author_id2num[row['author_id']] = idx
        graph_node_name['author'].append(row['author_name'])
    for idx, row in confs.iterrows():
        conf_id2num[row['conf_id']] = idx+num_author
        graph_node_name['conf'].append(row['conf'])
    for idx, row in papers.iterrows():
        paper_id2num[row['paper_id']] = idx+num_author+num_conf
        graph_node_name['paper'].append(row['paper_title'])
    for idx, row in terms.iterrows():
        term_id2num[row['term_id']] = idx+num_author+num_conf+num_paper
        graph_node_name['term'].append(row['term'])

    # save_pkl_file(save_prefix+"graph_node_name.pkl", graph_node_name)

    new_paper2author = []
    new_paper2term = []
    new_paper2conf = []
    for idx, row in paper_author.iterrows():
        src, dst = paper_id2num[row['paper_id']], author_id2num[row['author_id']]
        new_paper2author.append([src, dst])
    for idx, row in paper_term.iterrows():
        src, dst = paper_id2num[row['paper_id']], term_id2num[row['term_id']]
        new_paper2term.append([src, dst])
    for idx, row in paper_conf.iterrows():
        src, dst = paper_id2num[row['paper_id']], conf_id2num[row['conf_id']]
        new_paper2conf.append([src, dst])

    graph_data = {}
    edge_types = [('paper', 'was written by', 'author'), ('paper', 'was published in', 'term'), ('paper', 'was received by', 'conf')]
    reversed_edge_types = [('author', 'write', 'paper'), ('term', 'publish', 'paper'), ('conf', 'receive', 'paper')]
    src, dst = get_src_dst(new_paper2author, num_author+num_conf, 0)
    graph_data[edge_types[0]] = (th.tensor(src), th.tensor(dst))
    graph_data[reversed_edge_types[0]] = (th.tensor(dst), th.tensor(src))

    src, dst = get_src_dst(new_paper2term, num_author+num_conf, num_author+num_conf+num_paper)
    graph_data[edge_types[1]] = (th.tensor(src), th.tensor(dst))
    graph_data[reversed_edge_types[1]] = (th.tensor(dst), th.tensor(src))

    src, dst = get_src_dst(new_paper2conf, num_author+num_conf, num_author)
    graph_data[edge_types[2]] = (th.tensor(src), th.tensor(dst))
    graph_data[reversed_edge_types[2]] = (th.tensor(dst), th.tensor(src))

    g = dgl.heterograph(graph_data)
    g.nodes['author'].data['label'] = th.from_numpy(labels)
    # save_txt_file(save_prefix+"labels.txt", [str(i) for i in labels.tolist()])
    # save_graphs(save_prefix+"graph.bin", g)
    print("graph has been saved...")

    # # paper2author 0
    # # paper2term 1
    # # paper2conf 2
    # # author2paper 3
    # # term2paper 4
    # # conf2paper 5
    # edges = []
    # with open(save_prefix+'link.dat', 'w') as file:
    #     for i in range(len(new_paper2author)):
    #         src, dst = new_paper2author[i][0], new_paper2author[i][1]
    #         line = [str(src), str(dst), '0']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')

    #         line = [str(dst), str(src), '3']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')
    #     for i in range(len(new_paper2term)):
    #         src, dst = new_paper2term[i][0], new_paper2term[i][1]
    #         line = [str(src), str(dst), '1']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')

    #         line = [str(dst), str(src), '4']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')
    #     for i in range(len(new_paper2conf)):
    #         src, dst = new_paper2conf[i][0], new_paper2conf[i][1]
    #         line = [str(src), str(dst), '2']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')

    #         line = [str(dst), str(src), '5']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')
    # print("link.dat has been saved...")

    # with open(save_prefix+'node.dat', 'w') as file:
    #     for idx, row in author_label.iterrows():
    #         line = [str(idx), row['author_name'], '0']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')
    #     for idx, row in confs.iterrows():
    #         line = [str(idx+num_author), row['conf'], '1']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')
    #     for idx, row in papers.iterrows():
    #         line = [str(idx+num_author+num_conf), row['paper_title'], '2']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')
    #     for idx, row in terms.iterrows():
    #         line = [str(idx+num_author+num_conf+num_paper), row['term'], '3']
    #         line = '\t'.join(line)
    #         file.write(line + '\n')
    # print("node.dat has been saved...")