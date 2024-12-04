import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(root_path)

import time
import torch
import random
import joblib
import datetime
import numpy as np
from torch import nn
from copy import deepcopy
from sklearn import metrics
from argparse import ArgumentParser
from src.model import GraphModel, SharedEmbedding, FusionLayer
from src.data_config import data_args
from src.dataset import get_data_loader
from src.utils import setup_logger


def train_eval_stage_one(cate, loader, model, optimizer, loss_func, device):
    model.train() if cate == "train" else model.eval()
    preds, labels, loss_sum = [], [], 0.

    for i, graph in enumerate(loader):  # training under mini-batch, backpropagation under large batch

        graph = graph.to(device)
        targets = graph.y
        y = model(graph)
        loss = loss_func(y, targets)
        preds.append(y.max(dim=1)[1].data)
        labels.append(targets.data)

        if cate == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.data

    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = loss_sum / len(loader)
    acc = metrics.accuracy_score(labels, preds) * 100
    return loss, acc, preds, labels


def train_eval_stage_two(cate, loader, g1_model, g2_model, model, optimizer, loss_func, device):
    model.train() if cate == "train" else model.eval()
    preds, labels, loss_sum = [], [], 0.

    for i, graph in enumerate(loader):  # training under mini-batch, backpropagation under large batch

        graph = graph.to(device)
        targets = graph.y
        with torch.no_grad():
            y1 = g1_model(graph)
            y2 = g2_model(graph)
        y = model(y1, y2)
        loss = loss_func(y, targets)
        preds.append(y.max(dim=1)[1].data)
        labels.append(targets.data)

        if cate == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.data

    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = loss_sum / len(loader)
    acc = metrics.accuracy_score(labels, preds) * 100
    return loss, acc, preds, labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name.', type=str, default='R52')  # mr, ohsumed, 20ng, R8, R52
    parser.add_argument('--gpu', help='ID of available gpu.', type=int, default=0)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100)  # 150
    parser.add_argument('--batch_size', help='Size of batch for backpropagation.', type=int, default=512)  # 512, 64
    parser.add_argument('--input_dim', help='Dimension of input.', type=int, default=300)
    parser.add_argument('--hidden_dim', help='Number of units in hidden layer.', type=int, default=96)
    parser.add_argument('--num_layer', help='Number of graph layers.', type=int, default=2)
    parser.add_argument('--learning_rate', help='Initial learning rate.', type=float, default=0.005)
    parser.add_argument('--dropout', help='Dropout rate (1 - keep probability).', type=float, default=0.5)
    parser.add_argument('--weight_decay', help='Weight for L2 loss on embedding matrix.', type=float, default=1e-4)
    parser.add_argument('--not_freeze', help='Not freeze the param of embedding layer.', action='store_true')
    parser.add_argument('--rel_type', help='Relation type of single graph.', type=str, default='con')  # con, dep, occ
    parser.add_argument('--fuse_mode', help='Type of dual graph fusion.', type=str, default='gate')  # gate, concat
    parser.add_argument('--alpha', help='Weight of static fusion.', type=float, default=0.5)
    parser.add_argument('--share_gru', help='Share the params of GRU.', action='store_true')
    parser.add_argument('--fix_seed', help='Fix the random seed.', action='store_true')
    parser.add_argument('--seed', help='The random seed.', type=int, default=123)
    parser.add_argument('--log_dir', help='Log file path', type=str, default='log')

    arg = parser.parse_args()
    freeze = not arg.not_freeze

    if not os.path.exists(arg.log_dir):
        os.mkdir(arg.log_dir)

    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y%m%d%H%M%S")
    logger = setup_logger('DGAP', f'{arg.log_dir}/{arg.dataset}_switch_'
                                  f'{arg.fuse_mode}_{arg.share_gru}_{current_time_str}.log')
    logger.info(arg)
    logger.info(f"load arg.dataset: {arg.dataset}.")
    if arg.share_gru:
        logger.info(f"share gru.")
    else:
        logger.info(f"not share gru.")

    # random seed
    if arg.fix_seed:
        torch.manual_seed(arg.seed)
        torch.cuda.manual_seed(arg.seed)
        np.random.seed(arg.seed)
        random.seed(arg.seed)
        logger.info(f"fix seed: {arg.seed}.")
    else:
        logger.info(f"not fix seed.")

    num_classes = data_args[arg.dataset]['num_classes']
    (train_loader, valid_loader, test_loader), word2vec = get_data_loader(
        arg.dataset, arg.batch_size)
    logger.info(f"train size:{len(train_loader)}, valid size:{len(valid_loader)}, test size:{len(test_loader)}")
    num_words = len(word2vec) - 1  # index of special characters [PAD] in the vocab
    pos2idx = joblib.load(f"data/temp/{arg.dataset}.pos2index.pkl")
    num_pos = len(pos2idx)

    Device = torch.device(f'cuda:{arg.gpu}' if torch.cuda.is_available() else 'cpu')
    LossFunc = nn.CrossEntropyLoss()

    EmbedLayer = SharedEmbedding(num_words, arg.input_dim, word2vec=word2vec, freeze=freeze)
    EmbedLayer = EmbedLayer.to(Device)

    DepModel = GraphModel(num_words, num_classes, in_dim=arg.input_dim, hid_dim=arg.hidden_dim,
                          step=arg.num_layer, dropout=arg.dropout, num_pos=num_pos, rel_type='dep',
                          embed_layer=EmbedLayer)
    DepModel = DepModel.to(Device)
    DepOptimizer = torch.optim.Adam(DepModel.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay)

    # todo: test 1-layer
    ConModel = GraphModel(num_words, num_classes, in_dim=arg.input_dim, hid_dim=arg.hidden_dim,
                          step=arg.num_layer - 1, dropout=arg.dropout, num_pos=num_pos, rel_type='con',
                          embed_layer=EmbedLayer)
    ConModel = ConModel.to(Device)
    ConOptimizer = torch.optim.Adam(ConModel.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay)

    FusionModel = FusionLayer(num_classes, dropout=arg.dropout, alpha=arg.alpha, fuse_mode=arg.fuse_mode)
    FusionModel = FusionModel.to(Device)
    FusOptimizer = torch.optim.Adam(FusionModel.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay)

    logger.info("-" * 50)
    # todo: Add Weight Decay
    logger.info(f"params: [epoch_num={arg.epochs}, batch_size={arg.batch_size}, lr={arg.learning_rate}, "
                f"weight_decay={arg.weight_decay} , dropout={arg.dropout}]")
    logger.info("-" * 50)
    logger.info(DepModel)
    logger.info(ConModel)
    logger.info(FusionModel)
    logger.info("-" * 50)
    logger.info(f"Dataset: {arg.dataset}")

    best_acc_dep = 0.
    best_acc_test_dep = 0.
    best_net_dep = None

    best_acc_con = 0.
    best_acc_test_con = 0.
    best_net_con = None

    best_acc_fus = 0.
    best_acc_test_fus = 0.
    best_net_fus = None

    switch = False
    for epoch in range(arg.epochs):
        t1 = time.time()

        if switch is False:
            train_loss_dep, train_acc_dep, _, _ = train_eval_stage_one(
                "train", train_loader, DepModel, DepOptimizer, LossFunc, Device)
            valid_loss_dep, valid_acc_dep, _, _ = train_eval_stage_one(
                "valid", valid_loader, DepModel, DepOptimizer, LossFunc, Device)
            test_loss_dep, test_acc_dep, _, _ = train_eval_stage_one(
                "test", test_loader, DepModel, DepOptimizer, LossFunc, Device)

            if best_acc_dep < valid_acc_dep:  # save the best model on the validation set
                best_acc_dep = valid_acc_dep
                best_acc_test_dep = test_acc_dep
                best_net_dep = deepcopy(DepModel.state_dict())

            train_loss_con, train_acc_con, _, _ = train_eval_stage_one(
                "train", train_loader, ConModel, ConOptimizer, LossFunc, Device)
            valid_loss_con, valid_acc_con, _, _ = train_eval_stage_one(
                "valid", valid_loader, ConModel, ConOptimizer, LossFunc, Device)
            test_loss_con, test_acc_con, _, _ = train_eval_stage_one(
                "test", test_loader, ConModel, ConOptimizer, LossFunc, Device)

            if best_acc_con < valid_acc_con:  # save the best model on the validation set
                best_acc_con = valid_acc_con
                best_acc_test_con = test_acc_con
                best_net_con = deepcopy(ConModel.state_dict())

            cost = time.time() - t1

            logger.info(f"epoch={epoch + 1:03d}, time cost={cost:.2f}s; "
                        f"train dep:[{train_loss_dep:.4f}, {train_acc_dep:.2f}%], "
                        f"valid dep:[{valid_loss_dep:.4f}, {valid_acc_dep:.2f}%], "
                        f"test dep:[{test_loss_dep:.4f}, {test_acc_dep:.2f}%], "
                        f"best acc dep:[{best_acc_dep:.2f}% ({best_acc_test_dep:.2f}%)]; "
                        f"train con:[{train_loss_con:.4f}, {train_acc_con:.2f}%], "
                        f"valid con:[{valid_loss_con:.4f}, {valid_acc_con:.2f}%], "
                        f"test con:[{test_loss_con:.4f}, {test_acc_con:.2f}%], "
                        f"best acc con:[{best_acc_con:.2f}% ({best_acc_test_con:.2f}%)]")

        else:
            train_loss_fus, train_acc_fus, _, _ = train_eval_stage_two(
                "train", train_loader, DepModel, ConModel, FusionModel, FusOptimizer, LossFunc, Device)
            valid_loss_fus, valid_acc_fus, _, _ = train_eval_stage_two(
                "valid", valid_loader, DepModel, ConModel, FusionModel, FusOptimizer, LossFunc, Device)
            test_loss_fus, test_acc_fus, _, _ = train_eval_stage_two(
                "test", test_loader, DepModel, ConModel, FusionModel, FusOptimizer, LossFunc, Device)

            if best_acc_fus < valid_acc_fus:  # save the best model on the validation set
                best_acc_fus = valid_acc_fus
                best_acc_test_fus = test_acc_fus
                best_net_fus = deepcopy(FusionModel.state_dict())

            cost = time.time() - t1

            logger.info(f"epoch={epoch+1:03d}, time cost={cost:.2f}s; "
                        f"train:[{train_loss_fus:.4f}, {train_acc_fus:.2f}%], "
                        f"valid:[{valid_loss_fus:.4f}, {valid_acc_fus:.2f}%], "
                        f"test:[{test_loss_fus:.4f}, {test_acc_fus:.2f}%], "
                        f"best acc:[{best_acc_fus:.2f}% ({best_acc_test_fus:.2f}%)]")

        if epoch == int(arg.epochs * 0.9):
            switch = True
            DepModel.load_state_dict(best_net_dep)
            ConModel.load_state_dict(best_net_con)

    FusionModel.load_state_dict(best_net_fus)
    test_loss, test_acc, test_preds, test_labels = train_eval_stage_two(
        "test", test_loader, DepModel, ConModel, FusionModel, FusOptimizer, LossFunc, Device)

    logger.info("Test Precision, Recall and F1-Score...")
    logger.info(metrics.classification_report(test_labels, test_preds, digits=4))
    logger.info("Macro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='macro'))
    logger.info("Micro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='micro'))
