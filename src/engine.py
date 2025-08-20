import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from utils import *


class GeneGraphEngine:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)


    def loss(self, x1_train, x1_rec, x2_train, x2_rec, x2_pred, pred_adj, ori_adj ):
        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction="sum")
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction="sum")
        
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction="sum")

        # adj_loss = 
        return mse_x1 + mse_x2 +  mse_pert , \
                mse_x1, mse_x2, mse_pert 

    def train_epoch(self, train_loader,optimizer, adj):
        self.model.train()
        total_loss = 0
        for x1_train, x2_train, features, mol_id, cid, sig in train_loader:
            # 假设 batch = (drug_fp, gene_idx, labels)
            x1_train = x1_train.to(self.dev)
            x2_train = x2_train.to(self.dev)
            features = features.to(self.dev)
            if x1_train.shape[0] == 1:
                    continue
            optimizer.zero_grad()
            x1_rec, x2_pred, combine_adj = self.model(x1_train, features,adj)  # forward
            x2_mid = self.model.encoder_x2(x2_train)
            x2_rec = self.decoder_x2(x2_mid)
            
            loss,_, _, _ = self.loss(x1_train,x1_rec,x2_train,x2_rec,x2_pred,combine_adj,adj)
        
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)
    

    @torch.no_grad()
    def test(self, test_loader, adj, loss_item, metrics_func=None):
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

            x1_rec, x2_pred, combine_adj = self.model(x1_train, features,adj)  # forward
            x2_mid = self.model.encoder_x2(x2_train)
            x2_rec = self.decoder_x2(x2_mid)
            
            loss_ls= self.loss(x1_train,x1_rec,x2_train,x2_rec,x2_pred,combine_adj,adj)
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