import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from utils import *
from tqdm import tqdm
import math

class GeneGraphEngine:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.dev = device


    def loss(self, x1_train, x1_rec, x2_train, x2_rec, x2_pred, pred_adj, ori_adj, node_features ,epoch ):
        if epoch <100 :    
            loss_type = "mean"
        else:
            loss_type ="sum"
        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction=loss_type)
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction=loss_type)
        # print(x2_pred.min(), x2_pred.max(), x2_train.min(), x2_train.max())
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction=loss_type)
        B, N, d = node_features.shape   # 批大小、节点数、特征维度

        # 如果 ori_adj 是 [N, N]
        ori_adj = ori_adj.unsqueeze(0)              # 变成 [1, N, N]
        ori_adj = ori_adj.expand(B, -1, -1).contiguous()

        pred_adj_sub = pred_adj[:, :978, :978] if pred_adj.dim() == 3 else pred_adj[:978, :978]
        ori_adj_sub  = ori_adj[:, :978, :978]  if ori_adj.dim() == 3 else ori_adj[:978, :978]

        # 计算图差异损失
        # print(pred_adj_sub.device,ori_adj_sub.device)
        adj_loss = F.mse_loss(pred_adj_sub, ori_adj_sub, reduction='sum')
        # P = F.softmax(pred_adj_sub, dim=-1)
        # Q = F.softmax(ori_adj_sub, dim=-1)
        # L_adj_kl = batched_kl_div(P, Q)
        # # L_adj_kl = F.kl_div(normalize_adj_soft(pred_adj_sub).log(), normalize_adj_soft(ori_adj_sub), reduction='batchmean')
        # L_smooth = laplacian_smoothness_loss(ori_adj_sub, node_features[:,:-1])/(1024*978)

        # adj_loss =  L_adj_kl+ L_smooth L_adj_kl,L_smooth
        return mse_x1 + mse_x2 +  mse_pert + adj_loss,  \
                mse_x1, mse_x2, mse_pert , adj_loss,

    def loss_VIB(self, x1_train, x1_rec, x2_train, x2_rec, x2_pred, pred_adj, ori_adj, x1_mu, x1_std, x2_mu, x2_std, mu, std ,epoch ):
        # if epoch <100 :    
        #     loss_type = "mean"
        # else:
        loss_type ="sum"
        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction=loss_type)
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction=loss_type)
        # print(x2_pred.min(), x2_pred.max(), x2_train.min(), x2_train.max())
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction=loss_type)
        x1_KL_loss = -0.5 * (1 + 2 * x1_std.log() - x1_mu.pow(2) - x1_std.pow(2)).sum(1).mean().div(math.log(2))
        x2_KL_loss = -0.5 * (1 + 2 * x2_std.log() - x2_mu.pow(2) - x2_std.pow(2)).sum(1).mean().div(math.log(2))
        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        return mse_x1 + mse_x2  + x1_KL_loss+ x2_KL_loss ,  \
                mse_x1, mse_x2, mse_pert , x1_KL_loss, x2_KL_loss,KL_loss
    # 


    def train_epoch(self, train_loader,optimizer, adj, epoch,share_encoder = True):
        self.model.train()
        total_loss = 0
        train_size = 0
        for x1_train, x2_train, features, mol_id, cid, sig in tqdm(train_loader):
            # 假设 batch = (drug_fp, gene_idx, labels)
            x1_train = x1_train.to(self.dev)
            x2_train = x2_train.to(self.dev)
            features = features.to(self.dev)
            if x1_train.shape[0] == 1:
                    continue
            train_size += x1_train.shape[0]
            optimizer.zero_grad()
            x1_rec,x2_rec, x2_pred, combine_adj, (x1_mu,x1_std), (x2_mu,x2_std),(mu,std) = self.model(x1_train, features,adj,x2_train)  # forward
            # if share_encoder:
            #     x2_mid = self.model.Encoder_x1(x2_train)
            #     x2_rec = self.model.Decoder_x2(x2_mid)
            # else :
            #     x2_mid = self.model.Encoder_x2(x2_train)
            #     x2_rec = self.model.Decoder_x2(x2_mid)
            # print(x1_rec[1],x1_train[1])
            assert not torch.isnan(x1_train).any(), "x1_train contains NaN"
            assert not torch.isnan(x1_rec).any(), "x1_rec contains NaN"
            assert not torch.isnan(x2_train).any(), "x2_train contains NaN"
            assert not torch.isnan(x2_rec).any(), "x2_rec contains NaN"
            assert not torch.isnan(x2_pred).any(), "x2_pred contains NaN"
            loss,_1, _2, _3, _4, _5, _6  = self.loss_VIB(x1_train, x1_rec, x2_train, x2_rec, x2_pred, combine_adj, adj, x1_mu, x1_std, x2_mu, x2_std, mu, std, epoch)
            # print(loss,_1,_2,_3,_4)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  

        return total_loss / train_size
    

    @torch.no_grad()
    def test(self, test_loader, adj, loss_item, epoch, metrics_func=None,share_encoder=True):
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        
        self.model.eval()
        total_loss = 0
        test_size = 0
        all_scores, all_labels = [], []
        for x1_data, x2_data, features, mol_id, cid, sig in test_loader:
            x1_train = x1_data.to(self.dev)
            x2_train = x2_data.to(self.dev)
            features = features.to(self.dev)
            cid = np.array(list(cid))
            sig = np.array(list(sig))
            test_size += x1_data.shape[0]

            x1_rec,x2_rec,  x2_pred, combine_adj, (x1_mu,x1_std), (x2_mu,x2_std),(mu,std) = self.model(x1_train, features,adj, x2_train)  # forward
            # if share_encoder:

            #     x2_mid = self.model.Encoder_x1(x2_train)
            #     x2_rec = self.model.Decoder_x2(x2_mid)
            # else :
            #     x2_mid = self.model.Encoder_x2(x2_train)
            #     x2_rec = self.model.Decoder_x2(x2_mid)
            # loss_ls= self.loss(x1_train,x1_rec,x2_train,x2_rec,x2_pred,combine_adj, adj, node_features, epoch)
            loss_ls = self.loss_VIB(x1_train,x1_rec,x2_train,x2_rec,x2_pred,combine_adj, adj, x1_mu, x1_std, x2_mu, x2_std, mu, std , epoch)
            if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        test_dict[k] += loss_ls[idx].item()

            if metrics_func != None:
                metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(x1_data, x1_rec, x2_data, x2_rec, x2_pred, metrics_func=metrics_func)
                for k in metrics_dict.keys():
                    metrics_dict_all[k] += metrics_dict[k]
                for k in metrics_dict_ls.keys():
                    metrics_dict_all_ls[k] += metrics_dict_ls[k]

                metrics_dict_all_ls['cp_id'] += list(mol_id.numpy())
                metrics_dict_all_ls['cid'] += list(cid)
                metrics_dict_all_ls['sig'] += list(sig)


        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size
        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size
        return test_dict, metrics_dict_all, metrics_dict_all_ls

    @torch.no_grad()
    def predict(self, drug_fp, gene_idx):
        self.model.eval()
        drug_fp = drug_fp.to(self.device)
        gene_idx = gene_idx.to(self.device)
        scores = self.model(drug_fp, gene_idx)
        return scores.cpu()
    


    def eval_x_reconstruction(self, x1, x1_rec, x2, x2_rec, x2_pred, metrics_func=['pearson']):
        """
        Compute reconstruction evaluation metrics for x1_rec, x2_rec, x2_pred
        """
        x1_np = x1.data.cpu().numpy().astype(float)
        x2_np = x2.data.cpu().numpy().astype(float)
        x1_rec_np = x1_rec.data.cpu().numpy().astype(float)
        x2_rec_np = x2_rec.data.cpu().numpy().astype(float)
        x2_pred_np = x2_pred.data.cpu().numpy().astype(float)

        DEG_np = x2_np - x1_np
        DEG_rec_np = x2_rec_np - x1_np
        DEG_pert_np = x2_pred_np - x1_np

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)
        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])
                    metrics_dict['x1_rec_neg_' + m] += precision_neg
                    metrics_dict['x1_rec_pos_' + m] += precision_pos
                    metrics_dict_ls['x1_rec_neg_' + m] += [precision_neg]
                    metrics_dict_ls['x1_rec_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['x1_rec_' + m] += get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])
                    metrics_dict_ls['x1_rec_' + m] += [get_metric_func(m)(x1_np[i, :], x1_rec_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])
                    metrics_dict['x2_rec_neg_' + m] += precision_neg
                    metrics_dict['x2_rec_pos_' + m] += precision_pos
                    metrics_dict_ls['x2_rec_neg_' + m] += [precision_neg]
                    metrics_dict_ls['x2_rec_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['x2_rec_' + m] += get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])
                    metrics_dict_ls['x2_rec_' + m] += [get_metric_func(m)(x2_np[i, :], x2_rec_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])
                    metrics_dict['x2_pred_neg_' + m] += precision_neg
                    metrics_dict['x2_pred_pos_' + m] += precision_pos
                    metrics_dict_ls['x2_pred_neg_' + m] += [precision_neg]
                    metrics_dict_ls['x2_pred_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['x2_pred_' + m] += get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])
                    metrics_dict_ls['x2_pred_' + m] += [get_metric_func(m)(x2_np[i, :], x2_pred_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])
                    metrics_dict['DEG_rec_neg_' + m] += precision_neg
                    metrics_dict['DEG_rec_pos_' + m] += precision_pos
                    metrics_dict_ls['DEG_rec_neg_' + m] += [precision_neg]
                    metrics_dict_ls['DEG_rec_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['DEG_rec_' + m] += get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])
                    metrics_dict_ls['DEG_rec_' + m] += [get_metric_func(m)(DEG_np[i, :], DEG_rec_np[i, :])]

        for m in metrics_func:
            if m in ['precision10', 'precision20', 'precision50', 'precision100', 'precision200']:
                for i in range(x1_np.shape[0]):
                    precision_neg, precision_pos = get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])
                    metrics_dict['DEG_pred_neg_' + m] += precision_neg
                    metrics_dict['DEG_pred_pos_' + m] += precision_pos
                    metrics_dict_ls['DEG_pred_neg_' + m] += [precision_neg]
                    metrics_dict_ls['DEG_pred_pos_' + m] += [precision_pos]
            else:
                for i in range(x1_np.shape[0]):
                    metrics_dict['DEG_pred_' + m] += get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])
                    metrics_dict_ls['DEG_pred_' + m] += [get_metric_func(m)(DEG_np[i, :], DEG_pert_np[i, :])]
        return metrics_dict, metrics_dict_ls