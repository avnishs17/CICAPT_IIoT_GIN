import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv
from utils.utils import create_activation

class DeepGIN(nn.Module):
    def __init__(self,
                 n_dim,
                 e_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 mlp_layers,
                 activation,
                 feat_drop,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(DeepGIN, self).__init__()
        self.n_layers = n_layers
        self.gins = nn.ModuleList()

        last_activation = activation if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        for i in range(n_layers):
            if i == 0:
                in_dim = n_dim
            else:
                in_dim = hidden_dim

            if i == n_layers - 1:
                out_d = out_dim
            else:
                out_d = hidden_dim

            self.gins.append(DeepGINLayer(
                in_dim, e_dim, out_d, mlp_layers, feat_drop,
                activation if i < n_layers - 1 else last_activation,
                residual if i < n_layers - 1 else last_residual,
                norm=norm if i < n_layers - 1 else last_norm
            ))

    def forward(self, g, h, return_hidden=False):
        hidden_list = []
        for i in range(self.n_layers):
            h = self.gins[i](g, h)
            hidden_list.append(h)
        if return_hidden:
            return h, hidden_list
        else:
            return h

class DeepGINLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 e_dim,
                 out_dim,
                 mlp_layers,
                 feat_drop,
                 activation,
                 residual,
                 norm):
        super(DeepGINLayer, self).__init__()
        self.apply_func = MLP(in_dim, out_dim, out_dim, mlp_layers)
        self.gin_layer = GINConv(self.apply_func, 'mean', learn_eps=True)
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = create_activation(activation)
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)

    def forward(self, g, feat):
        h = self.feat_drop(feat)
        h = self.gin_layer(g, h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = h + self.res_fc(feat)
        if self.norm is not None:
            h = self.norm(h)
        return h

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(MLP, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.activation = F.relu

        if num_layers == 1:
            self.linear_layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.linear_layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.linear_layers):
            h = layer(h)
            if i != len(self.linear_layers) - 1:
                h = self.activation(h)
        return h
    

#improve this model without chanign input and output of it