
import imp
import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from utils import *
from Gaug import *




class GeneGraph(torch.nn.Module):
    def __init__(self, n_genes, n_embedd, n_latent, n_en_hidden, n_de_hidden, features_dim, features_embed_dim, **kwargs):
        """ The parse of the GeneGraph """
        super(GeneGraph,self).__init__()
        self.n_genes = n_genes      # num of Gene
        self.n_gene_embedd = n_embedd
        self.n_latent = n_latent     # 
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.features_dim = features_dim
        self.feat_embed_dim = features_embed_dim

        # Encoder
        self.gene_embedding = nn.Parameter(torch.randn(self.n_genes, self.n_gene_embedd))

        self.pre_mlp = nn.Sequential(
            nn.Linear( self.n_gene_embedd, self.n_en_hidden),
            nn.ReLU(),
            nn.Linear(self.n_en_hidden, self.n_gene_embedd)
        )


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_gene_embedd, nhead=6, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.n_gene_embedd, self.n_en_hidden),
            nn.ReLU(),
            nn.Linear(self.n_en_hidden, 1)  # 每个基因重建回标量
        )

        # Fusion network
        self.fusion_network = VGAE(dim_feats, dim_h, dim_z, activation,  gae=gae)    # VGAE 为边预测函数





    def encode(self, x):
        H = x.unsqueeze(-1) * self.gene_embedding.unsqueeze(0)  
        H = self.pre_mlp(H)
        H = self.transformer(H)
        return H
    def decode(self, H):
        H_flat = H.view(-1, H.size(-1))
        # 通过共享 MLP 解码
        x_hat_flat = self.decoder(H_flat)
        # 还原形状 (batch, num_genes)
        x_hat = x_hat_flat.view(H.size(0), H.size(1))
        return x_hat
    
    def forward(self, x):
        H = self.encode(x)
        x_hat = self.decode(H)
        return x_hat, H
