from .gin import DeepGIN
from utils.utils import create_norm
from functools import partial
from itertools import chain
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random

def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    mlp_layers = args.mlp_layers
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    model = DeepGINAutoencoder(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        mlp_layers=mlp_layers,
        activation="prelu",  # Pass as a string
        feat_drop=0.2,
        residual=True,
        mask_rate=mask_rate,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l
    )
    return model

class DeepGINAutoencoder(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, mlp_layers, activation,
                 feat_drop, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(DeepGINAutoencoder, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        # build encoder
        self.encoder = DeepGIN(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
            mlp_layers=mlp_layers,
            activation=activation,
            feat_drop=feat_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )

        # build decoder for attribute prediction
        self.decoder = DeepGIN(
            n_dim=hidden_dim,
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            out_dim=n_dim,
            n_layers=1,
            mlp_layers=mlp_layers,
            activation=activation,
            feat_drop=feat_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(hidden_dim * n_layers, hidden_dim, bias=False)

        # setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token

        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        # Feature Reconstruction
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(pre_use_g, rep)
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)

        # Structural Reconstruction
        threshold = min(10000, g.num_nodes())

        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
            g.device)
        loss += self.recon_loss(y_pred, y)
        return loss

    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])