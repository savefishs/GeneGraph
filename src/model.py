
import imp
from pyparsing import Combine
import torch
import torch.nn.functional as F
from torch import layer_norm, nn, optim
import copy
from collections import defaultdict
from utils import *
from GNNmodel import *


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
        self.activation = nn.ReLU()
        self.gnnlayer_type = 'gat'
        self.n_layers = 6
        

        # Encoder
        self.gene_embedding_x1 = nn.Parameter(torch.randn(self.n_genes, self.n_gene_embedd))

        self.gene_embedding_x2 = nn.Parameter(torch.randn(self.n_genes, self.n_gene_embedd))


        encoder = [
            nn.Linear(self.n_gene_embedd, self.n_en_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_en_hidden, self.n_gene_embedd),
            nn.ReLU(),
        ]

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_gene_embedd, 
            nhead=6, 
            batch_first=True
        )
        encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=2))

        # 包装为两个 encoder
        self.encoder_x1 = nn.Sequential(*copy.deepcopy(encoder))
        self.encoder_x2 = nn.Sequential(*copy.deepcopy(encoder))

        # Decoder
        decoder = [
            nn.Linear(self.n_gene_embedd, self.n_de_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_de_hidden, 1)  # 每个基因映射回标量
        ]

        # 包装为两个 decoder
        self.decoder_x1 = nn.Sequential(*copy.deepcopy(decoder))
        self.decoder_x2 = nn.Sequential(*copy.deepcopy(decoder))

        # drug feature complce
        self.net = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 1024),
                nn.layer_norm(1024)
            )


        # feature fusion and get new feature. 

        ## drug-gene predict 
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.n_gene_embedd + self.n_latent, 256),
            nn.ReLU(),
            nn.Linear(256, 1)        
        )

        self.Graph_predicit = VGAE(self.n_gene_embedd, self.n_en_hidden,  self.n_latent, self.activation,  gae=False)    # VGAE 为边预测函数

        self.GNN = GNNEncoder(self.n_gene_embedd, self.n_en_hidden , self.n_layers , self.activation, dropout=0.05, gnnlayer_type=self.gnnlayer_type)

        # self.Graph_deal = GAug_model(self.n_gene_embedd, self.n_en_hidden, self.n_latent, self.n_layers, self.activation, dropout=0.05, gnnlayer_type=self.gnnlayer_type, device='GPU')


    def encoder_x1(self, x):
        H = x.unsqueeze(-1) * self.gene_embedding_x1.unsqueeze(0)  
        H = self.encoder_x1(H)
        return H

    def encoder_x2(self, x):
        H = x.unsqueeze(-1) * self.gene_embedding_x2.unsqueeze(0)  
        H = self.encoder_x2(H)
        return H

    def decoder_x1(self, H):
        H_flat = H.view(-1, H.size(-1))
        # 通过共享 MLP 解码
        x_hat_flat = self.decoder(H_flat)
        # 还原形状 (batch, num_genes)
        x_hat = x_hat_flat.view(H.size(0), H.size(1))
        return x_hat

    def decoder_x2(self, H):
        H_flat = H.view(-1, H.size(-1))
        # 通过共享 MLP 解码
        x_hat_flat = self.decoder(H_flat)
        # 还原形状 (batch, num_genes)
        x_hat = x_hat_flat.view(H.size(0), H.size(1))
        return x_hat

    def edge_predict(self, drug_fp, gene_ids=None):
        """
        drug_fp: [B, drug_fp_dim] Morgan fingerprint
        gene_ids: [N] 要考虑的基因id (如果None就用所有基因)
        """
        drug_emb = self.drug_encoder(drug_fp)     # [B, latent_dim]
        if gene_ids is None:
            gene_embs = self.gene_emb             # [n_genes, gene_emb_dim]
        else:
            gene_embs = self.gene_emb[gene_ids]   # [N, gene_emb_dim]

        # 扩展药物embedding，和每个基因拼接
        B, d = drug_emb.shape
        N = gene_embs.size(0)
        drug_expand = drug_emb.unsqueeze(1).expand(B, N, d)  # [B, N, latent_dim]
        gene_expand = gene_embs.unsqueeze(0).expand(B, N, -1) # [B, N, gene_emb_dim]

        pair = torch.cat([drug_expand, gene_expand], dim=-1) # [B, N, latent+gene_emb_dim]

        scores = self.edge_predictor(pair).squeeze(-1) # [B, N]
        return scores  # 越大越可能有边 
    
    def adj_combine(self, edge_predicted, adj):

        B, N = edge_predicted.shape
        
        device = edge_predicted.device

        gene_adj = adj.to_dense().unsqueeze(0).expand(B, N, N)

        gene_drug = edge_predicted.unsqueeze(-1)   # (B, N, 1)
        drug_gene = edge_predicted.unsqueeze(1)    # (B, 1, N)

    # 药物-药物对角补 0
        drug_drug = torch.zeros(B, 1, 1, device=device)

        # 拼接成 (B, N+1, N+1)
        top = torch.cat([gene_adj, gene_drug], dim=2)      # (B, N, N+1)
        bottom = torch.cat([drug_gene, drug_drug], dim=2)  # (B, 1, N+1)
        new_adj = torch.cat([top, bottom], dim=1)          # (B, N+1, N+1)

        return new_adj
    
    def forward(self, x, features, adj):# -> tuple[Any, Any]:
        H = self.encoder_x1(x)
        features = self.net(features)
        predice_edge = self.edge_predict(features)
        new_adj = self.adj_combine(predice_edge,adj)
        combine_adj =  self.Graph_predicit(new_adj,features)

        if self.sample_type == 'edge':
            adj_new = self.sample_adj_edge(combine_adj, adj, self.alpha)
        elif self.sample_type == 'add_round':
            adj_new = self.sample_adj_add_round(combine_adj, adj, self.alpha)
        elif self.sample_type == 'rand':
            adj_new = self.sample_adj_random(combine_adj)
        elif self.sample_type == 'add_sample':
            if self.alpha == 1:
                adj_new = self.sample_adj(combine_adj)
            else:
                adj_new = self.sample_adj_add_bernoulli(combine_adj, adj, self.alpha)
        new_combine_adj = self.normalize_adj(adj_new)
        new_features = self.GNN(new_combine_adj,features)

        x2_pred = self.decoder_x2(new_features)


        x1_rec = self.decoder_x1(H)
        return x1_rec, x2_pred, combine_adj


    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_bernoulli(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_add_round(self, adj_logits, adj_orig, alpha):
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = RoundNoGradient.apply(edge_probs)
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def sample_adj_random(self, adj_logits):
        adj_rand = torch.rand(adj_logits.size())
        adj_rand = adj_rand.triu(1)
        adj_rand = torch.round(adj_rand)
        adj_rand = adj_rand + adj_rand.T
        return adj_rand

    def sample_adj_edge(self, adj_logits, adj_orig, change_frac):
        adj = adj_orig.to_dense() if adj_orig.is_sparse else adj_orig
        n_edges = adj.nonzero().size(0)
        n_change = int(n_edges * change_frac / 2)
        # take only the upper triangle
        edge_probs = adj_logits.triu(1)
        edge_probs = edge_probs - torch.min(edge_probs)
        edge_probs = edge_probs / torch.max(edge_probs)
        adj_inverse = 1 - adj
        # get edges to be removed
        mask_rm = edge_probs * adj
        nz_mask_rm = mask_rm[mask_rm>0]
        if len(nz_mask_rm) > 0:
            n_rm = len(nz_mask_rm) if len(nz_mask_rm) < n_change else n_change
            thresh_rm = torch.topk(mask_rm[mask_rm>0], n_rm, largest=False)[0][-1]
            mask_rm[mask_rm > thresh_rm] = 0
            mask_rm = CeilNoGradient.apply(mask_rm)
            mask_rm = mask_rm + mask_rm.T
        # remove edges
        adj_new = adj - mask_rm
        # get edges to be added
        mask_add = edge_probs * adj_inverse
        nz_mask_add = mask_add[mask_add>0]
        if len(nz_mask_add) > 0:
            n_add = len(nz_mask_add) if len(nz_mask_add) < n_change else n_change
            thresh_add = torch.topk(mask_add[mask_add>0], n_add, largest=True)[0][-1]
            mask_add[mask_add < thresh_add] = 0
            mask_add = CeilNoGradient.apply(mask_add)
            mask_add = mask_add + mask_add.T
        # add edges
        adj_new = adj_new + mask_add
        return adj_new

    def normalize_adj(self, adj):
        if self.gnnlayer_type == 'gcn':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
            adj = D_norm @ adj @ D_norm
        elif self.gnnlayer_type == 'gat':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
        elif self.gnnlayer_type == 'gsage':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj