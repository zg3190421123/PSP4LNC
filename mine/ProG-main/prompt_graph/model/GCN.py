import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from prompt_graph.utils import act

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean'):
        super().__init__()
        GraphConv = GCNConv

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([
                GraphConv(input_dim, hid_dim),
                GraphConv(hid_dim, out_dim)
            ])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio

        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # Define normalization layers
        self.layer_norms = torch.nn.ModuleList([torch.nn.LayerNorm(hid_dim) for _ in range(num_layer - 1)])
        if num_layer > 1:
            self.layer_norms.append(torch.nn.LayerNorm(out_dim))

    def forward(self, x, edge_index, batch=None, prompt=None, prompt_type=None):
        h_list = [x]
        for idx, (conv, ln) in enumerate(zip(self.conv_layers[0:-1], self.layer_norms)):
            x = conv(x, edge_index)
            x = ln(x)
            x = F.relu(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        if len(self.layer_norms) > len(self.conv_layers) - 1:
            x = self.layer_norms[-1](x)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        if batch is None:
            return node_emb
        else:
            if prompt_type == 'Gprompt':
                node_emb = prompt(node_emb)
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb


    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()