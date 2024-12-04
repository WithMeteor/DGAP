import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn
import torch.nn.functional as fn


class GRUUint(nn.Module):

    def __init__(self, hid_dim, act, bias=True):
        super(GRUUint, self).__init__()
        self.act = act
        self.lin_z0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_z1 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_r0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_r1 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_h0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_h1 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, a):
        z = (self.lin_z0(a) + self.lin_z1(x)).sigmoid()
        r = (self.lin_r0(a) + self.lin_r1(x)).sigmoid()
        h = self.act((self.lin_h0(a) + self.lin_h1(x * r)))
        return h * z + x * (1 - z)


class GraphLayer(gnn.MessagePassing):

    def __init__(self, hid_dim, dropout=0.5,
                 act=torch.relu, step=2, rel='dep', gru=None):
        super(GraphLayer, self).__init__(aggr='add')
        self.step = step
        self.rel = rel

        if gru is not None:
            self.gru = gru  # shared gru for dual graph
        else:
            self.gru = GRUUint(hid_dim, act=act)  # individual gru for each graph

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, g):
        for i in range(self.step):
            if self.rel == 'dep':
                a = self.propagate(edge_index=g.edge_index_dep, x=x, edge_attr=self.dropout(g.edge_attr_dep))
            else:
                a = self.propagate(edge_index=g.edge_index_con, x=x, edge_attr=self.dropout(g.edge_attr_con))
            x = self.gru(x, a)

        return x

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.unsqueeze(-1)

    def update(self, inputs):
        return inputs


class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim, att_dim, dropout=0.5, num_pos=17):
        super(ReadoutLayer, self).__init__()
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.act2 = torch.tanh
        self.bi_lstm = nn.LSTM(in_dim, in_dim, bidirectional=True, batch_first=True)
        self.proj_layer = nn.Linear(in_dim * 2, att_dim, bias=True)
        self.att_layer = nn.Linear(att_dim, 1, bias=False)
        self.pos_bias = nn.Linear(num_pos, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, out_dim, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, p, mask):

        # # Attention Weight Sum - Words
        x, _ = self.bi_lstm(x)
        emb = self.proj_layer(x)
        ett = self.act1(self.att_layer(emb)) + self.act2(self.pos_bias(p))
        att = self.mask_softmax(ett, mask)
        x = att * x  # x.shape: (batch_size, graph_node_num, hidden_dim)
        x = torch.sum(x, dim=1)

        # x.shape: (batch_size, hidden_dim)
        x = self.mlp(x)
        return x

    @staticmethod
    def mask_softmax(x, mask):
        mask_data = x.masked_fill(mask.logical_not(), -1e9)
        return fn.softmax(mask_data, dim=1)


class SharedEmbedding(nn.Module):
    def __init__(self, num_words, embed_dim, word2vec=None, freeze=True):
        super(SharedEmbedding, self).__init__()
        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, embed_dim, num_words)
        else:
            self.embed = nn.Embedding.from_pretrained(
                torch.from_numpy(word2vec).float(), freeze, num_words)

    def forward(self, x):
        return self.embed(x)


class GraphModel(nn.Module):
    def __init__(self, num_words, num_classes, in_dim=300, hid_dim=96, att_dim=64, step=2, dropout=0.5,
                 num_pos=17, rel_type='dep', embed_layer=None):
        super(GraphModel, self).__init__()
        if embed_layer is None:
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = embed_layer
        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hid_dim, bias=True)
        )
        self.gnn_layer = GraphLayer(hid_dim, act=torch.tanh, dropout=dropout, step=step,
                                    rel=rel_type, gru=GRUUint(hid_dim, act=torch.tanh))
        self.read_layer = ReadoutLayer(hid_dim, num_classes, att_dim=att_dim, dropout=dropout, num_pos=num_pos)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g):
        x = self.embed(g.x)
        p = self.graph2batch(g.pos, g.length)
        # The use of stop word masks did not improve the performance
        mask = self.get_mask(g)

        x = torch.tanh(self.encode(x))
        x = self.gnn_layer(x, g)  # x_dep.shape: (batch_node_num, hidden_dim)
        # g.length.shape: (batch_size)
        x = self.graph2batch(x, g.length)  # combine the graph of a mini-batch and restore the batch
        y = self.read_layer(x, p, mask)
        return y

    def get_mask(self, g, mask=None):
        if mask is None:
            mask = pad_sequence([torch.ones(ln) for ln in g.length], batch_first=True).unsqueeze(-1)
        else:
            mask = self.graph2batch(mask, g.length).unsqueeze(-1)
        if g.x.is_cuda:
            mask = mask.cuda(device=g.x.device)
        return mask

    @staticmethod
    def graph2batch(x, length_list):
        x_list = []
        for graph_len in length_list:
            x_list.append(x[:graph_len])
            x = x[graph_len:]
        x = pad_sequence(x_list, batch_first=True)
        return x


class FusionLayer(nn.Module):

    def __init__(self, hid_dim, dropout=0.5, alpha=0.5, fuse_mode='gate'):
        super(FusionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if fuse_mode == 'gate':
            self.gate = nn.Linear(hid_dim * 2, 1, bias=False)
        elif fuse_mode == 'atten':
            self.att = nn.Linear(hid_dim, 1, bias=False)
        elif fuse_mode == 'concat':
            self.lin = nn.Linear(hid_dim * 2, hid_dim, bias=False)
        self.initial_alpha = alpha  # hyper-parameters for static fusion
        self.register_buffer('alpha', torch.Tensor([alpha]))
        self.fuse_mode = fuse_mode
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.alpha.data.fill_(self.initial_alpha)

    def forward(self, x_dep, x_con):

        if self.fuse_mode == 'gate':
            # # Gate Mechanism
            x = torch.cat((x_dep, x_con), dim=-1)
            h = torch.sigmoid(self.gate(x))
            x = h * x_dep + (1 - h) * x_con  # (batch_node_num, hidden_dim)

        elif self.fuse_mode == 'atten':
            # # Atten Mechanism
            w_dep = fn.elu(self.att(x_dep))
            w_con = fn.elu(self.att(x_con))
            h = fn.softmax(torch.cat((w_dep, w_con), dim=-1), dim=-1)
            x = torch.unsqueeze(h[:, 0], -1) * x_dep + torch.unsqueeze(h[:, 1], -1) * x_con

        elif self.fuse_mode == 'concat':
            # Concat Mechanism
            x = self.lin(torch.cat((x_dep, x_con), dim=-1))

        else:  # mode == 'static'
            # Sum Mechanism
            x = self.alpha * x_dep + (1 - self.alpha) * x_con

        # x.shape: (batch_node_num, hidden_dim)
        return x
