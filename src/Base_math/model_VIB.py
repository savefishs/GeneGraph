
from hmac import new
import imp
from json import decoder
from pyparsing import Combine
import torch
import torch.nn.functional as F
from torch import layer_norm, nn, optim
from torch.autograd import Variable
import copy
from collections import defaultdict
from .utils import *
from .VIB_backbone import *
from .VIB_layers import * 
from torch_geometric.data import Data

class GeneGraph_VIB(torch.nn.Module):
    def __init__(self, n_genes, n_embedd, n_latent, n_en_hidden, n_de_hidden, features_dim, features_embed_dim, gene_emb_tensor, args):
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
        self.hidden_dim = 512
        self.backbone = 'GCN'
        self.graph_type = 'KNN' # "KNN" ,"epsilonNN", "prob"
        self.top_k = 50
        self.epsilon =0.3
        self.num_pers = 8
        self.metric_type = "attention"
        self.feature_denoise = False
        self.device = "cuda"
        self.IB_size = 64
        self.beta = 0.00001

        self.n_layers = 3
        self.args=args
        self.dropout = args.dropout
        self.init_w = args.init_w

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
            nn.Linear(self.n_en_hidden, self.IB_size*2),
        ]
        self.ln = nn.LayerNorm(self.IB_size*2)

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
        decoder = nn.Sequential(
            nn.Linear(self.IB_size, self.n_de_hidden),
            nn.BatchNorm1d(self.n_de_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_de_hidden, self.n_genes),  # 映射到 N 个基因
            nn.ReLU()
        )


        # 包装为两个 decoder
        self.decoder_x1 = nn.Sequential(*copy.deepcopy(decoder))
        self.decoder_x2 = nn.Sequential(*copy.deepcopy(decoder))
        
        # drug feature complce
        self.net = nn.Sequential(
                nn.Linear(2304, self.n_gene_embedd),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.n_gene_embedd, self.IB_size*2),
                nn.LayerNorm(self.IB_size*2)
            )


        # feature fusion and get new feature. 

        ## drug-gene predict 


        self.graph_learner = GraphLearner(input_size=self.IB_size*2, hidden_size=self.hidden_dim,
                                          graph_type=self.graph_type, top_k=self.top_k,
                                          epsilon=self.epsilon, num_pers=self.num_pers, metric_type=self.metric_type,
                                          feature_denoise=self.feature_denoise, device=None)

        # self.GNN = GNNEncoder(dim_feats=self.n_gene_embedd, dim_h=self.n_en_hidden ,dim_out=self.n_gene_embedd ,n_layers=self.n_layers , activation=self.activation, dropout=0.05, gnnlayer_type=self.gnnlayer_type)

        if self.backbone == "GCN":
            self.GNN = myGCN(self.args, in_dim=self.IB_size*2, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)
        elif self.backbone == "GIN":
            self.GNN = myGIN(self.args, in_dim=self.IB_size*2, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)
        elif self.backbone == "GAT":
            self.GNN = myGAT(self.args, in_dim=self.IB_size*2, out_dim=self.IB_size*2,
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
        H = gate * self.gene_embedding_x1.weight.unsqueeze(0) # 
        H = self.encoder_x1(H)
        return H

    def Encoder_x2(self, x):
        gate = torch.sigmoid(self.gate(x.unsqueeze(-1)))  # (B, n_genes, embed_dim)
        H = gate * self.gene_embedding_x1.weight.unsqueeze(0) # 
        H = self.encoder_x1(H)
        return H

    def Decoder_x1(self, H):
      
        x_hat_flat = self.decoder_x1(H)
        return x_hat_flat

    def Decoder_x2(self, H):
        x_hat_flat = self.decoder_x2(H)
        return x_hat_flat


    def learn_graph(self, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        # node_features: (B, N, F)
        # init_adj:      (B, N, N)

        new_feature, new_adj = self.graph_learner(node_features)  # 支持 batch

        if graph_skip_conn in (0.0, None):
            # if graph_include_self:
            #     I = torch.eye(new_adj.size(-1), device=new_adj.device).unsqueeze(0)  # (1, N, N)
            #     new_adj = new_adj + I
            new_adj_final = new_adj
        else:
            new_adj_final = graph_skip_conn * init_adj + (1 - graph_skip_conn) * new_adj
        
        if graph_include_self:
            I = torch.eye(new_adj_final.size(-1), device=new_adj_final.device).unsqueeze(0)
            new_adj_final = new_adj_final + I

        return new_feature, new_adj

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def forward(self, x1, features, adj,x2=None):# -> tuple[Any, Any]:
        new_graph_list = []
        B = x1.size(0)
        H1 = self.Encoder_x1(x1)
        # features = self.net(features)
        # fusion_features  = torch.cat([H1, features[:, None, :].expand(-1, self.n_genes, -1)], dim=-1)
        # # fusion_features = H + features[:, None, :].expand(-1, self.n_genes, -1)
        # # print(fusion_features.shape)

        # ## X1 graph
        # x1_cat = H1.view(B * self.n_genes, -1)  # [B*N, D]
        # x1_batch_cat = torch.arange(B, device=H1.device).repeat_interleave(self.n_genes)  # [B*N]
        # x1_graph_embs = global_mean_pool(x1_cat, x1_batch_cat)  # [B, D]
        # x1_mu = x1_graph_embs[:, :self.IB_size]
        # x1_std = F.softplus(x1_graph_embs[:, self.IB_size:] - self.IB_size, beta=1)
        # new_x1_graph_embs = self.reparametrize_n(x1_mu, x1_std, B)
        # print(new_x1_graph_embs.shape)

        # if x2 != None:
        H2 = self.Encoder_x2(x2)
        #     x2_cat = H2.view(B * self.n_genes, -1)  # [B*N, D]
        #     x2_batch_cat = torch.arange(B, device=H2.device).repeat_interleave(self.n_genes)  # [B*N]
        #     x2_graph_embs = global_mean_pool(x2_cat, x2_batch_cat)  # [B, D]
        #     x2_mu = x2_graph_embs[:, :self.IB_size]
        #     x2_std = F.softplus(x2_graph_embs[:, self.IB_size:] - self.IB_size, beta=1)
        #     new_x2_graph_embs = self.reparametrize_n(x2_mu, x2_std, B)
        #     x2_rec = self.Decoder_x2(new_x2_graph_embs)

        ## for x1， x2 graph

        new_graph_list_x1 = []
        new_graph_list_x2 = []
        for i in range(B):
            features_x1 = H1[i]
            features_x2 = H2[i]
            # graph generate
            new_features_x1 ,x1_adj = self.learn_graph(node_features=features_x1,graph_skip_conn=self.args.graph_skip_conn,
            graph_include_self=self.args.graph_include_self)
            new_features_x2 ,x2_adj = self.learn_graph(node_features=features_x2,graph_skip_conn=self.args.graph_skip_conn,
            graph_include_self=self.args.graph_include_self)
            # add to Data
            new_edge_index_x1, new_edge_attr_x1 = dense_to_sparse(x1_adj)
            new_graph_x1 = Data(x=new_features_x1, edge_index=new_edge_index_x1, edge_attr=new_edge_attr_x1)
            new_graph_list_x1.append(new_graph_x1)

            new_edge_index_x2, new_edge_attr_x2 = dense_to_sparse(x2_adj)
            new_graph_x2 = Data(x=new_features_x2, edge_index=new_edge_index_x2, edge_attr=new_edge_attr_x2 )
            new_graph_list_x2.append(new_graph_x2)

        X1_loader = DataLoader(new_graph_list_x1, batch_size=len(new_graph_list_x1))
        X2_loader =  DataLoader(new_graph_list_x2, batch_size=len(new_graph_list_x2))
        X1_batch = next(iter(X1_loader))
        X2_batch = next(iter(X2_loader))

        x1_node_embs, _ = self.GNN(X1_batch.x, X1_batch.edge_index ,edge_weight=X1_batch.edge_attr)
        x2_node_embs, _ = self.GNN(X2_batch.x, X2_batch.edge_index ,edge_weight=X2_batch.edge_attr)
        
        x1_graph_embs = global_mean_pool(x1_node_embs, X1_batch.batch)
        x1_mu = x1_graph_embs[:, :self.IB_size]
        x1_std = F.softplus(x1_graph_embs[:, self.IB_size:] - self.IB_size, beta=1)
        x1_new_graph_embs = self.reparametrize_n(x1_mu, x1_std, B)

        x2_graph_embs = global_mean_pool(x2_node_embs, X2_batch.batch)
        x2_mu = x2_graph_embs[:, :self.IB_size]
        x2_std = F.softplus(x2_graph_embs[:, self.IB_size:] - self.IB_size, beta=1)
        x2_new_graph_embs = self.reparametrize_n(x2_mu, x2_std, B)

        x1_rec = self.Decoder_x1(x1_new_graph_embs)
        x2_rec = self.Decoder_x2(x2_new_graph_embs)
        return x1_rec, x2_rec, (x1_mu,x1_std), (x2_mu,x2_std)

        x, edge_index = adj.x.to(device), adj.edge_index.to(device)
        for i in range(B):
            fusion_features_N = fusion_features[i]
            raw_adj = to_dense_adj(edge_index)[0]
            new_features,new_adj = self.learn_graph(node_features=fusion_features_N,graph_skip_conn=self.args.graph_skip_conn,
            graph_include_self=self.args.graph_include_self,
            init_adj=raw_adj)
            new_edge_index, new_edge_attr = dense_to_sparse(new_adj)
            new_graph = Data(x=new_features, edge_index=new_edge_index, edge_attr=new_edge_attr)
            new_graph_list.append(new_graph)

        loader = DataLoader(new_graph_list, batch_size=len(new_graph_list))
        batch_data = next(iter(loader))
        node_embs, _ = self.GNN(batch_data.x, batch_data.edge_index ,edge_weight=batch_data.edge_attr)


        graph_embs = global_mean_pool(node_embs, batch_data.batch)
        mu = graph_embs[:, :self.IB_size]
        std = F.softplus(graph_embs[:, self.IB_size:] - self.IB_size, beta=1)
        new_graph_embs = self.reparametrize_n(mu, std, B)


        x2_pred = self.Decoder_x2(new_graph_embs)
        

        x1_rec = self.Decoder_x1(new_x1_graph_embs)
        # return x1_rec,x2_rec
        return x1_rec, x2_rec, x2_pred, new_adj, (x1_mu,x1_std), (x2_mu,x2_std),(mu,std)


    





