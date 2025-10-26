import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from Base_math.utils import *
from tqdm import tqdm
import math

class GeneGraphEngine:
    def __init__(self, model, beta ,device='cpu'):
        self.model = model.to(device)
        self.dev = device
        self.beta = beta
        # self.model_type = model_type
        

    def loss(self, cls_pred, cls_label):
   
        ys = cls_label.detach().cpu().numpy()  # 转 numpy
        min_s = np.min(ys)
        eps = 1e-8
        weights = np.log(ys - min_s + np.e + eps)   # 和 get_loss_weight 一样
        weights = torch.tensor(weights, dtype=torch.float32, device=cls_label.device)

        # 计算加权 MSE
        mse = (weights * (cls_pred.squeeze() - cls_label) ** 2).mean()
        # reg_loss = F.mse_loss(cls_pred.squeeze(), cls_label, reduction="mean")

        return mse
    
    def evaluate_regression(self, y_pred, y_true):
        """
        y_pred : torch.Tensor 或 np.array, 模型预测值
        y_true : torch.Tensor 或 np.array, 真实标签
        返回 MSE 和 PCC
        """
        # 转为 numpy
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy().squeeze()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy().squeeze()

        # MSE
        mse = np.mean((y_pred - y_true) ** 2)

        # PCC
        pcc, _ = pearsonr(y_pred, y_true)

        return mse, pcc
    
    def get_loss_weight(T_train) :
        ys = T_train.syn_ans.squeeze().tolist()
        min_s = np.amin(ys)
        loss_weight = np.log(ys - min_s + np.e)
        return list(loss_weight)

    def train_epoch(self, train_loader,optimizer, adj= None, epoch =None,share_encoder = True):
        self.model.train()
        total_loss = 0
        train_size = 0
        for cell_base, features_A, features_B, label in tqdm(train_loader):
            # 假设 batch = (drug_fp, gene_idx, labels)
            cell_base = cell_base.to(self.dev)
            features_A = features_A.to(self.dev)
            features_B = features_B.to(self.dev)
            label = label.to(self.dev)
            if cell_base.shape[0] == 1:
                    continue
            train_size += cell_base.shape[0]
            optimizer.zero_grad()

            pred_score = self.model(cell_base, features_A, features_A)

            loss = self.loss(pred_score, label)
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
        for cell_base, features_A, features_B, label in test_loader:
            cell_base = cell_base.to(self.dev)
            features_A = features_A.to(self.dev)
            features_B = features_B.to(self.dev)
            label = label.to(self.dev)
            test_size += cell_base.shape[0]

            pred_score = self.model(cell_base, features_A, features_A)
            all_scores.append(pred_score.detach().cpu())
            all_labels.append(label.detach().cpu())

        all_scores = torch.cat(all_scores, dim=0).numpy().squeeze()
        all_labels = torch.cat(all_labels, dim=0).numpy().squeeze()

    # 计算 MSE 和 PCC
        mse, pcc = self.evaluate_regression(all_scores, all_labels)
        
        return mse, pcc

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