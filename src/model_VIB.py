
from hmac import new
import imp
from pyparsing import Combine
import torch
import torch.nn.functional as F
from torch import layer_norm, nn, optim
import copy
from collections import defaultdict
from utils import *
from VIB_backbone import *
from VIB_layers import * 


class GeneGraph_VIB(torch.nn.Module):
    def __init__(self, n_genes, n_embedd, n_latent, n_en_hidden, n_de_hidden, features_dim, features_embed_dim, gene_emb_tensor, **kwargs):
        """ The parse of the GeneGraph """
        super(GeneGraph_VIB,self).__init__()
        self.n_genes = n_genes      # num of Gene
        self.n_gene_embedd = n_embedd
        self.n_latent = n_latent     # 
        self.n_en_hidden = n_en_hidden[0]
        self.n_de_hidden = n_de_hidden[0]
        self.features_dim = features_dim
        self.feat_embed_dim = features_embed_dim
        self.activation = nn.ReLU()
        
        #graph part
        self.hidden_dim = 128
        self.gnnlayer_type = 'gcn'
        self.graph_type = 'KNN' # "KNN" ,"epsilonNN", "prob"
        self.top_k = 50
        self.epsilon =0.3
        self.num_pers = 16
        self.metric_type = "attention"
        self.feature_denoise = True
        self.device = "cuda"

        self.n_layers = 3
        self.dropout = kwargs["dropout"]
        self.init_w = kwargs['init_w']

        self.sample_type = 'rand'
        self.alpha=1
        self.gene_emb_tensor = gene_emb_tensor
        

        # Encoder
        self.gene_embedding_x1 = nn.Embedding(self.n_genes, self.n_gene_embedd)

        self.gene_embedding_x2 = nn.Embedding(self.n_genes, self.n_gene_embedd)

        self.gate = nn.Linear(1, self.n_gene_embedd)

        encoder = [
            nn.Linear(self.n_gene_embedd, self.n_en_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_en_hidden, self.n_gene_embedd),
        ]
        self.ln = nn.LayerNorm(self.n_gene_embedd)

        # TransformerEncoder
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.n_gene_embedd, 
        #     nhead=4, 
        #     batch_first=True
        # )
        # encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=2))

        # 包装为两个 encoder
        self.encoder_x1 = nn.Sequential(*copy.deepcopy(encoder))
        self.encoder_x2 = nn.Sequential(*copy.deepcopy(encoder))

        # Decoder
        decoder = [
            nn.Linear(self.n_gene_embedd, self.n_de_hidden),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_de_hidden, 1),  # 每个基因映射回标量
            nn.ReLU()
        ]

        # 包装为两个 decoder
        self.decoder_x1 = nn.Sequential(*copy.deepcopy(decoder))
        self.decoder_x2 = nn.Sequential(*copy.deepcopy(decoder))
        
        # drug feature complce
        self.net = nn.Sequential(
                nn.Linear(2304, self.n_gene_embedd),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.n_gene_embedd, self.n_gene_embedd),
                nn.LayerNorm(self.n_gene_embedd)
            )


        # feature fusion and get new feature. 

        ## drug-gene predict 
        # self.drug_encoder = nn.Sequential(
        #     nn.Linear(1024,512),
        #     nn.ReLU()
        # )

        self.edge_predictor = nn.Sequential(
            nn.Linear(self.n_gene_embedd * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.Graph_predicit = GraphLearner(input_size=self.n_gene_embedd, hidden_size=self.hidden_dim,
                                          graph_type=self.graph_type, top_k=self.top_k,
                                          epsilon=self.epsilon, num_pers=self.num_pers, metric_type=self.metric_type,
                                          feature_denoise=self.feature_denoise, device=None)

        # self.GNN = GNNEncoder(dim_feats=self.n_gene_embedd, dim_h=self.n_en_hidden ,dim_out=self.n_gene_embedd ,n_layers=self.n_layers , activation=self.activation, dropout=0.05, gnnlayer_type=self.gnnlayer_type)

        if self.backbone == "GCN":
            self.GNN = myGCN(self.args, in_dim=self.n_gene_embedd, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)
        elif self.backbone == "GIN":
            self.GNN = myGIN(self.args, in_dim=self.n_gene_embedd, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)
        elif self.backbone == "GAT":
            self.GNN = myGAT(self.args, in_dim=self.n_gene_embedd, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)

        # self.Graph_deal = GAug_model(self.n_gene_embedd, self.n_en_hidden, self.n_latent, self.n_layers, self.activation, dropout=0.05, gnnlayer_type=self.gnnlayer_type, device='GPU')

        if self.init_w:
            self.encoder_x1.apply(self._init_weights)
            self.decoder_x1.apply(self._init_weights)

            self.encoder_x2.apply(self._init_weights)
            self.decoder_x2.apply(self._init_weights)
            if gene_emb_tensor is not None:
                self.gene_embedding_x1.weight.data = gene_emb_tensor.clone()
                self.gene_embedding_x2.weight.data = gene_emb_tensor.clone()



            
        
    def _init_weights(self, layer):
        """ Initialize weights of layer with Xavier uniform"""
        if type(layer)==nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    def Encoder_x1(self, x):
        ## simple encoder
        # H = x.unsqueeze(-1) * self.gene_embedding_x1.unsqueeze(0)  
        ## gate encoder
        gate = torch.sigmoid(self.gate(x.unsqueeze(-1)))  # (B, n_genes, embed_dim)
        H = gate * self.gene_embedding_x1.weight.unsqueeze(0) # gating 控制
        H_res = H
        H = F.gelu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H = H + H_res
        H = self.ln(H)
        return H

    def Encoder_x2(self, x):
        gate = torch.sigmoid(self.gate(x.unsqueeze(-1)))  # (B, n_genes, embed_dim)
        H = gate * self.gene_embedding_x2.weight.unsqueeze(0)     # gating 控制
        H_res = H
        H = F.gelu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H = H + H_res
        H = self.ln(H)
        return H

    def Decoder_x1(self, H):
        H_flat = H.view(-1, H.size(-1))
        # 通过共享 MLP 解码
        x_hat_flat = self.decoder_x1(H_flat)
        # 还原形状 (batch, num_genes)
        x_hat = x_hat_flat.view(H.size(0), H.size(1))
        return x_hat

    def Decoder_x2(self, H):
        H_flat = H.reshape(-1, H.size(-1))
        # 通过共享 MLP 解码
        x_hat_flat = self.decoder_x2(H_flat)
        # 还原形状 (batch, num_genes)

        x_hat = x_hat_flat.view(H.size(0), H.size(1))
        return x_hat

    def edge_predict(self, drug_fp, gene_emb):
        """
        drug_fp: [B, drug_fp_dim] Morgan fingerprint
        gene_ids: [N] 要考虑的基因id (如果None就用所有基因)
        """
        # drug_emb = self.drug_encoder(drug_fp)     # [B, latent_dim]

           
        # 扩展药物embedding，和每个基因拼接
        B, d = drug_fp.shape
        _,N,_ =gene_emb.shape
       
        drug_expand = drug_fp.unsqueeze(1).expand(B, N, d)  # [B, N, latent_dim]

        # gene_expand = gene_embs.unsqueeze(0).expand(B, N, -1) # [B, N, gene_emb_dim]

        pair = torch.cat([drug_expand, gene_emb], dim=-1) # [B, N, latent+gene_emb_dim]

        pred_edges = self.edge_predictor(pair).squeeze(-1) # [B, N]

        # K = 50  # 每个节点保留的边数
        # topk_vals, topk_idx = torch.topk(pred_edges.abs(), K, dim=1)
        # mask = torch.zeros_like(pred_edges)
        # mask.scatter_(1, topk_idx, 1.0)
        # pred_edges = pred_edges * mask
        # sign = pred_edges.sign()        # 保存正负方向
        # abs_vals = pred_edges.abs()
        # norm_vals = abs_vals / (abs_vals.sum(dim=1, keepdim=True) + 1e-8)  # 防止除0
        # pred_edges = norm_vals * sign
        return pred_edges  # 越大越可能有边 
    
    def adj_combine(self, edge_predicted, adj):

        B, N = edge_predicted.shape
        
        device = edge_predicted.device

        # gene_adj = adj.to_dense().to(device).unsqueeze(0).expand(B, N, N)
        gene_adj = adj.to(device).unsqueeze(0).expand(B, N, N)
        gene_drug = edge_predicted.unsqueeze(-1)   # (B, N, 1)
        drug_gene = edge_predicted.unsqueeze(1)    # (B, 1, N)

    # 药物-药物对角补 0
        drug_drug = torch.zeros(B, 1, 1, device=device)

        # 拼接成 (B, N+1, N+1)
        top = torch.cat([gene_adj, gene_drug], dim=2)      # (B, N, N+1)
        bottom = torch.cat([drug_gene, drug_drug], dim=2)  # (B, 1, N+1)
        new_adj = torch.cat([top, bottom], dim=1)          # (B, N+1, N+1)

        return new_adj
    

    def learn_graph(self, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        # node_features: (B, N, F)
        # init_adj:      (B, N, N)

        new_feature, new_adj = self.graph_learner(node_features)  # 支持 batch

        if graph_skip_conn in (0.0, None):
            if graph_include_self:
                I = torch.eye(new_adj.size(-1), device=new_adj.device).unsqueeze(0)  # (1, N, N)
                new_adj = new_adj + I
        else:
            new_adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * new_adj

        return new_feature, new_adj

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def forward(self, x, features, adj):# -> tuple[Any, Any]:
        B = x.size(0)
        H = self.Encoder_x1(x)
        features = self.net(features)

        fusion_features = torch.cat([H,features.unsqueeze(1)], dim=1) 


        new_features,new_adj = self.learn_graph(node_features=fusion_features,graph_skip_conn=self.args.graph_skip_conn,
        graph_include_self=self.args.graph_include_self,
        init_adj=adj)

        all_edge_index = []
        all_edge_attr = []
        all_x = []
        all_batch = []

        for i in range(B):
            ei, ea = dense_to_sparse(new_adj[i])
            ei = ei + i * self.n_genes  # shift node index per graph
            all_edge_index.append(ei)
            all_edge_attr.append(ea)
            all_x.append(new_features[i])
            all_batch.append(torch.full((self.n_genes,), i, dtype=torch.long, device=device))

        x_cat = torch.cat(all_x, dim=0)  # (B*N, F)
        edge_index_cat = torch.cat(all_edge_index, dim=1)
        edge_attr_cat = torch.cat(all_edge_attr, dim=0)
        batch_cat = torch.cat(all_batch, dim=0)

        node_embs, _ = self.gnn(x_cat, edge_index_cat, edge_weight=edge_attr_cat)

        graph_embs = global_mean_pool(node_embs, batch_cat)
        mu = graph_embs[:, :self.IB_size]
        std = F.softplus(graph_embs[:, self.IB_size:] - self.IB_size, beta=1)
        new_graph_embs = self.reparametrize_n(mu, std, B)


        x2_pred = self.Decoder_x2(new_features[:,:-1])
        

        x1_rec = self.Decoder_x1(H)
        return x1_rec, x2_pred, new_adj,fusion_features


    





