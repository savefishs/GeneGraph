import gc
import math
import time
from xml.dom import HierarchyRequestErr
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from sklearn.metrics import f1_score
import pyro


class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.gcn_base = GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False)
        self.gcn_mean = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)
        self.gcn_logstd = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)

    
    def forward(self, adj, features):
        """
        adj: (N, N) or (B, N, N)
        features: (N, F) or (B, N, F)
        return: adj_logits (N, N) or (B, N, N)
        """
        hidden = self.gcn_base(adj, features)      # (B, N, H)
        hidden = torch.tanh(hidden)
        mean   = self.gcn_mean(adj, hidden)        # (B, N, Z)
        # print(mean[1,1])
        if self.gae:
            Z = mean
        else:
            logstd = self.gcn_logstd(adj, hidden)  # (B, N, Z)
            logstd = torch.clamp(logstd, min=-10, max=10)

            # print(logstd[1,1])
            
            gaussian_noise = torch.randn_like(mean)
            Z = gaussian_noise * torch.exp(logstd) + mean

        adj_logits = torch.bmm(Z, Z.transpose(1, 2))  # (B, N, N)
        adj_prob = torch.sigmoid(adj_logits)
        return adj_prob
    


class GNNEncoder(nn.Module):
    """ GNN as node representation model """
    def __init__(self, dim_feats, dim_h, dim_out, n_layers, activation, dropout, gnnlayer_type='gcn'):
        super(GNNEncoder, self).__init__()
        heads = [1] * (n_layers + 1)
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        elif gnnlayer_type == 'gsage':
            gnnlayer = SAGELayer
        elif gnnlayer_type == 'gat':
            gnnlayer = GATLayer
            if dim_feats in (50, 745, 12047): # hard coding n_heads for large graphs
                heads = [2] * n_layers + [1]
            else:
                heads = [8] * n_layers + [1]
            dim_h = int(dim_h / 8)
            dropout = 0.6
            activation = F.elu

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(gnnlayer(dim_feats, dim_h, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(gnnlayer(dim_h*heads[i], dim_h, heads[i+1], activation, dropout))
        # output embedding layer
        self.layers.append(gnnlayer(dim_h*heads[-2], dim_out, heads[-1], None, dropout))

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        return h   # 返回节点 embedding (N x dim_out)
    


class GCNLayer(nn.Module):
    """ one layer of GCN, supports batch input (B, N, N) """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.b = nn.Parameter(torch.FloatTensor(output_dim)) if bias else None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if param.dim() == 2:   # weight
                nn.init.xavier_uniform_(param)
            else:                 # bias
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        """
        adj: (N, N) or (B, N, N)
        h:   (N, F) or (B, N, F)
        """
        if self.dropout is not None:
            h = self.dropout(h)

        if adj.dim() == 2:  # 单图 (N, N)
            x = h @ self.W         # (N, F_out)
            x = adj @ x            # (N, F_out)
        else:  # 批量图 (B, N, N)
            x = h @ self.W         # (B, N, F_out)
            x = torch.bmm(adj, x)  # (B, N, F_out)

        if self.b is not None:
            x = x + self.b

        if self.activation:
            x = self.activation(x)
        return x


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g
    

class SAGELayer(nn.Module):
    """ one layer of GraphSAGE with GCN aggregator, supports batch input (B, N, N) """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(SAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if param.dim() == 2:   # weight
                nn.init.xavier_uniform_(param)
            else:                 # bias
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        """
        adj: (N, N) or (B, N, N)
        h:   (N, F) or (B, N, F)
        """
        if self.dropout:
            h = self.dropout(h)

        if adj.dim() == 2:  # 单图
            x = adj @ h                  # (N, F)
            x = self.linear_neigh(x)     # (N, F_out)
        else:  # 批量
            x = torch.bmm(adj, h)        # (B, N, F)
            x = self.linear_neigh(x)     # (B, N, F_out)

        if self.activation:
            x = self.activation(x)
        return x

class GATLayer(nn.Module):
    """ one layer of GAT, supports batch input (B, N, N) """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.n_heads = n_heads
        self.attn_l = nn.Linear(output_dim, n_heads, bias=False)
        self.attn_r = nn.Linear(output_dim, n_heads, bias=False)
        self.attn_drop = nn.Dropout(p=0.6)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.b = nn.Parameter(torch.FloatTensor(output_dim)) if bias else None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if param.dim() == 2:   # weight
                nn.init.xavier_uniform_(param)
            else:                 # bias
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        """
        adj: (N, N) or (B, N, N)
        h:   (N, F) or (B, N, F)
        """
        if self.dropout:
            h = self.dropout(h)

        # (B, N, F_out)
        if h.dim() == 2:   # 单图 (N, F)
            h = h.unsqueeze(0)    # 变成 (1, N, F)
            adj = adj.unsqueeze(0)
        B, N, _ = h.size()

        x = torch.matmul(h, self.W)  # (B, N, F_out)

        # 注意力计算
        el = self.attn_l(x)  # (B, N, n_heads)
        er = self.attn_r(x)  # (B, N, n_heads)

        # 广播相加得到 (B, N, N, n_heads)
        attn = el.unsqueeze(2) + er.unsqueeze(1)
        attn = F.leaky_relu(attn, negative_slope=0.2)

        # mask 非边
        attn = attn.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        attn = F.softmax(attn, dim=2)    # (B, N, N, n_heads)
        attn = self.attn_drop(attn)

        # 消息传递
        x = torch.einsum("bijh,bjf->bihf", attn, x)  # (B, N, n_heads, F_out)
        if self.b is not None:
            x = x + self.b

        if self.activation:
            x = self.activation(x)

        if self.n_heads > 1:
            x = x.reshape(B, N, -1)  # 拼接 heads
        else:
            x = x.squeeze(2)  # (B, N, F_out)

        if B == 1:  # 如果输入是单图，还原成 (N, F)
            x = x.squeeze(0)

        return x


class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g
    
