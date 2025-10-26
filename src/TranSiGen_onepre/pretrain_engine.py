import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from Base_math.utils import *
from tqdm import tqdm
import math

class GeneGraphEngine:
    def __init__(self, model, beta, device='cpu'):
        self.model = model.to(device)
        self.dev = device
        self.beta = beta
        

    def loss(self, x1_train, x1_rec, mu1, logvar1, x2_train, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred):

        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction="sum")
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction="sum")
        mse_pert = F.mse_loss(x2_pred , x2_train, reduction="sum")

        kld_x1 = -0.5 * torch.sum(1. + logvar1 - mu1.pow(2) - logvar1.exp(), )
        kld_x2 = -0.5 * torch.sum(1. + logvar2 - mu2.pow(2) - logvar2.exp(), )
        kld_pert = -0.5 * torch.sum(1. + logvar_pred - mu_pred.pow(2) - logvar_pred.exp(), )
        # kld_pert = -0.5 * torch.sum(1 + (logvar_pred - logvar2) - ((mu_pred - mu2).pow(2) + logvar_pred.exp()) / logvar2.exp(), )


        return mse_x1 + mse_x2 + mse_pert + self.beta * kld_x1 + self.beta * kld_x2 + self.beta * kld_pert, \
               mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert

    def train_epoch(self, train_loader,optimizer, adj= None, epoch =None,share_encoder = True):
        self.model.train()
        total_loss = 0
        train_size = 0
        for cell_base, drug_pert, features in tqdm(train_loader):
            # 假设 batch = (drug_fp, gene_idx, labels)
            cell_base = cell_base.to(self.dev)
            drug_pert = drug_pert.to(self.dev)
            features = features.to(self.dev)
            if cell_base.shape[0] == 1:
                    continue
            train_size += cell_base.shape[0]
            optimizer.zero_grad()

            x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.model(cell_base, features)

            z2, mu2, logvar2 = self.model.encode_x2(drug_pert)
            x2_rec = self.model.decode_x2(z2)

            loss, mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert = self.loss(cell_base, x1_rec, mu1, logvar1, drug_pert, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  

        return total_loss / train_size
    

    @torch.no_grad()
    def test(self, test_loader, adj= None, epoch =None, loss_item = None, metrics_func=None,share_encoder=True):
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        
        self.model.eval()
        total_loss = 0
        test_size = 0
        all_scores, all_labels = [], []
        for cell_base, drug_pert, features in test_loader:
            cell_base = cell_base.to(self.dev)
            drug_pert = drug_pert.to(self.dev)
            features = features.to(self.dev)
            test_size += cell_base.shape[0]

            x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.model(cell_base, features)

            z2, mu2, logvar2 = self.model.encode_x2(drug_pert)
            x2_rec = self.model.decode_x2(z2)


            loss_ls = self.loss(cell_base, x1_rec, mu1, logvar1, drug_pert, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred)


            if loss_item != None:
                for idx, k in enumerate(loss_item):
                    test_dict[k] += loss_ls[idx].item()

            # if metrics_func != None:
            #     metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(cell_base, x1_rec, drug_pert, x2_rec, x2_pred, metrics_func=metrics_func)
            #     for k in metrics_dict.keys():
            #         metrics_dict_all[k] += metrics_dict[k]
            #     for k in metrics_dict_ls.keys():
            #         metrics_dict_all_ls[k] += metrics_dict_ls[k]

            #     metrics_dict_all_ls['cp_id'] += list(mol_id.numpy())
            #     metrics_dict_all_ls['cid'] += list(cid)
            #     metrics_dict_all_ls['sig'] += list(sig)


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