import torch
import joblib
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn.functional import one_hot
from src.data_config import data_args


def split_train_valid_test(data, train_size, valid_part=0.1):
    train_data = data[:train_size]
    test_data = data[train_size:]
    random.shuffle(train_data)
    valid_size = round(valid_part * train_size)
    valid_data = train_data[:valid_size]
    train_data = train_data[valid_size:]
    return train_data, valid_data, test_data


def get_data_loader(dataset, batch_size):
    # param
    train_size = data_args[dataset]["train_size"]

    # load data
    len_input = joblib.load(f"data/graph/{dataset}.len.inputs.pkl")
    len_graphs_con = joblib.load(f"data/graph/{dataset}.len.graphs.con.pkl")
    len_graphs_dep = joblib.load(f"data/graph/{dataset}.len.graphs.dep.pkl")

    word_input = np.load(f"data/graph/{dataset}.input.word.npy")
    pos_input = np.load(f"data/graph/{dataset}.input.pos.npy")
    word_mask = np.load(f"data/graph/{dataset}.mask.word.npy")

    graphs_con = np.load(f"data/graph/{dataset}.graphs.con.npy")
    weight_con = np.load(f"data/graph/{dataset}.weight.con.npy")
    graphs_dep = np.load(f"data/graph/{dataset}.graphs.dep.npy")
    weight_dep = np.load(f"data/graph/{dataset}.weight.dep.npy")

    word2vec = np.load(f"data/temp/{dataset}.word2vec.npy")
    targets = np.load(f"data/temp/{dataset}.targets.npy")
    pos2idx = joblib.load(f"data/temp/{dataset}.pos2index.pkl")
    pos_num = len(pos2idx)

    data = []
    for x, y, p, m, li, lo, ld, eo, wo, ed, wd in tqdm(list(zip(
            word_input, targets, pos_input, word_mask,
            len_input, len_graphs_con, len_graphs_dep,
            graphs_con, weight_con, graphs_dep, weight_dep
    )), ascii=True):
        x = torch.tensor(x[:li], dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        p = one_hot(torch.tensor(p[:li], dtype=torch.long), pos_num).float()
        m = torch.tensor(m[:li], dtype=torch.bool)
        lens = torch.tensor(li, dtype=torch.long)
        edge_index_con = torch.tensor(np.array([e[:lo] for e in eo]), dtype=torch.long)
        edge_attr_con = torch.tensor(wo[:lo], dtype=torch.float)
        edge_index_dep = torch.tensor(np.array([e[:ld] for e in ed]), dtype=torch.long)
        edge_attr_dep = torch.tensor(wd[:ld], dtype=torch.float)
        data.append(Data(x=x, y=y, pos=p, m=m, length=lens,
                         edge_index_con=edge_index_con, edge_attr_con=edge_attr_con,
                         edge_index_dep=edge_index_dep, edge_attr_dep=edge_attr_dep))

    # split
    train_data, valid_data, test_data = split_train_valid_test(data, train_size, valid_part=0.1)
    return [DataLoader(data, batch_size=batch_size, shuffle=True) for data in
            [train_data, valid_data, test_data]], word2vec

