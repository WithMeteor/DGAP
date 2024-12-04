import json
import joblib
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from collections import Counter
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name.', type=str, default='mr')  # mr, ohsumed, 20ng, R8, R52
    args = parser.parse_args()
    return args


# normalize
def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized


# calculate edge weight & normalize adjacency matrix
def cal_graph_weight(edges, nrow, ncol):
    edge_count = Counter(edges).items()
    row = [x for (x, y), c in edge_count]
    col = [y for (x, y), c in edge_count]
    weight = [c for _, c in edge_count]

    # normalize adjacency matrix
    adj = sp.csr_matrix((weight, (row, col)), shape=(nrow, ncol))
    adj_norm = normalize_adj(adj)
    weight_norm = [adj_norm[x][y] for (x, y), c in edge_count]

    return row, col, weight_norm


def pad_seq(seq, pad_len, pad_elem=0):
    if len(seq) > pad_len:
        return seq[:pad_len]
    return seq + [pad_elem] * (pad_len - len(seq))


def construct_graph(dataset, parse_list, word2index, pos2index):

    # build graph
    pos_input = []  # the universe part-of-speech type of word nodes
    word_mask = []  # the mask of word nodes
    word_input = []  # the input word nodes
    graphs_con = []  # topology and weight of constituency edges
    graphs_dep = []  # topology and weight of dependency edges
    graphs_occ = []  # topology and weight of co-occurrence edges

    print("Build graph:")
    for doc_id in tqdm(range(len(parse_list)), ascii=True):

        depen_edge_list = parse_list[doc_id]['depen_edge']
        const_edge_list = parse_list[doc_id]['const_edge']
        token_node_list = parse_list[doc_id]['token_list']
        unpos_type_list = parse_list[doc_id]['unpos_type']
        token_mask_list = parse_list[doc_id]['token_mask']

        assert len(token_node_list) == len(unpos_type_list) == len(token_mask_list)
        sent_len = len(token_node_list)

        con_edges, dep_edges = [], []
        occ_edges = []

        for const_tuple in const_edge_list:  # add constituency edges within the constituency
            source_idx = const_tuple[0]
            target_idx = const_tuple[1]
            con_edges.append((source_idx, target_idx))
            con_edges.append((target_idx, source_idx))

        for depen_tuple in depen_edge_list:  # add dependency edges
            source_idx = depen_tuple[0]
            target_idx = depen_tuple[2]
            dep_edges.append((source_idx, target_idx))
            dep_edges.append((target_idx, source_idx))

        # todo: co-occurrence edges are just useful for ablation study
        for i in range(len(token_node_list)):  # add co-occurrence edges
            target_idx = i
            for j in range(i - 1, i + 2):
                if i != j and 0 <= j < len(token_node_list):
                    source_idx = j
                    occ_edges.append((source_idx, target_idx))
            # add co-occurrence node pairs with the distance of 2
            if i + 2 < len(token_node_list):
                source_idx = i + 2
                occ_edges.append((source_idx, target_idx))
                occ_edges.append((target_idx, source_idx))

        id_word_list = [word2index[w] for w in token_node_list]
        word_input.append(id_word_list)

        id_pos_list = [pos2index[p] for p in unpos_type_list]
        pos_input.append(id_pos_list)

        word_mask.append(token_mask_list)

        # calculate constituency edge weight
        row, col, weight_norm = cal_graph_weight(con_edges, sent_len, sent_len)
        graphs_con.append([row, col, weight_norm])

        # calculate dependency edge weight
        row, col, weight_norm = cal_graph_weight(dep_edges, sent_len, sent_len)
        graphs_dep.append([row, col, weight_norm])

        # calculate co-occurrence edge weight
        row, col, weight_norm = cal_graph_weight(occ_edges, sent_len, sent_len)
        graphs_occ.append([row, col, weight_norm])

    # The number of nodes and edges of each graph is recorded here
    # to preserve the boundary for restoration after padding.
    len_input = [len(e) for e in word_input]
    len_graphs_con = [len(x) for x, y, c in graphs_con]
    len_graphs_dep = [len(x) for x, y, c in graphs_dep]
    len_graphs_occ = [len(x) for x, y, c in graphs_occ]

    # padding input
    pad_len_input = max(len_input)  # Maximum number of nodes in text graph
    pad_len_con_graph = max(len_graphs_con)  # Maximum number of constituency edges
    pad_len_dep_graph = max(len_graphs_dep)  # Maximum number of dependency edges
    pad_len_occ_graph = max(len_graphs_occ)  # Maximum number of co-occurrence edges

    # padding to save as numpy-array
    word_mask_pad = [pad_seq(e, pad_len_input, 0) for e in word_mask]
    pos_input_pad = [pad_seq(e, pad_len_input, len(pos2index)) for e in pos_input]
    word_input_pad = [pad_seq(e, pad_len_input, len(word2index)) for e in word_input]
    graphs_con_pad = [[pad_seq(ee, pad_len_con_graph) for ee in e] for e in graphs_con]
    graphs_dep_pad = [[pad_seq(ee, pad_len_dep_graph) for ee in e] for e in graphs_dep]
    graphs_occ_pad = [[pad_seq(ee, pad_len_occ_graph) for ee in e] for e in graphs_occ]

    word_mask_pad = np.array(word_mask_pad)  # word_mask_pad.shape: doc_num, max_node_num
    pos_input_pad = np.array(pos_input_pad)  # pos_input_pad.shape: doc_num, max_node_num
    word_input_pad = np.array(word_input_pad)  # word_input_pad.shape: doc_num, max_node_num
    weight_con_pad = np.array([c for x, y, c in graphs_con_pad])  # weight_con_pad.shape: doc_num, max_edge_num
    graphs_con_pad = np.array([[x, y] for x, y, c in graphs_con_pad])  # graphs_con_pad.shape: doc_num, max_edge_num, 2
    weight_dep_pad = np.array([c for x, y, c in graphs_dep_pad])  # weight_dep_pad.shape: doc_num, max_edge_num
    graphs_dep_pad = np.array([[x, y] for x, y, c in graphs_dep_pad])  # graphs_dep_pad.shape: doc_num, max_edge_num, 2
    weight_occ_pad = np.array([c for x, y, c in graphs_occ_pad])  # weight_occ_pad.shape: doc_num, max_edge_num
    graphs_occ_pad = np.array([[x, y] for x, y, c in graphs_occ_pad])  # graphs_occ_pad.shape: doc_num, max_edge_num, 2

    # save
    joblib.dump(len_input, f"data/graph/{dataset}.len.inputs.pkl")
    joblib.dump(len_graphs_con, f"data/graph/{dataset}.len.graphs.con.pkl")
    joblib.dump(len_graphs_dep, f"data/graph/{dataset}.len.graphs.dep.pkl")

    np.save(f"data/graph/{dataset}.input.pos.npy", pos_input_pad)
    np.save(f"data/graph/{dataset}.input.word.npy", word_input_pad)
    np.save(f"data/graph/{dataset}.mask.word.npy", word_mask_pad)

    np.save(f"data/graph/{dataset}.graphs.con.npy", graphs_con_pad)
    np.save(f"data/graph/{dataset}.weight.con.npy", weight_con_pad)
    np.save(f"data/graph/{dataset}.graphs.dep.npy", graphs_dep_pad)
    np.save(f"data/graph/{dataset}.weight.dep.npy", weight_dep_pad)

    np.save(f"data/graph/{dataset}.graphs.occ.npy", graphs_occ_pad)
    np.save(f"data/graph/{dataset}.weight.occ.npy", weight_occ_pad)
    joblib.dump(len_graphs_occ, f"data/graph/{dataset}.len.graphs.occ.pkl")


def main():
    args = get_args()
    dataset = args.dataset

    with open('data/temp/{}.parse.json'.format(dataset), 'r', encoding='utf-8') as file:
        parse_list = json.load(file)

    word2index = joblib.load(f"data/temp/{dataset}.word2index.pkl")
    pos2index = joblib.load(f"data/temp/{dataset}.pos2index.pkl")

    print('Start constructing graph of dataset:', dataset)

    construct_graph(dataset, parse_list, word2index, pos2index)


if __name__ == '__main__':
    main()
