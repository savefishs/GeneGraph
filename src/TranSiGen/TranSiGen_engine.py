from cProfile import label
import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from Base_math.utils import *
from tqdm import tqdm
import math


class TranSiGen_engine:
    def __init__(self, model, device='cpu',model_type='reg'):
        self.model = model.to(device)
        self.dev = device
        self.model_type = model_type

    def loss_reg(self, pred_label,  mu_pred, logvar_pred ):


        kld_pert = -0.5 * torch.sum(1. + logvar_pred - mu_pred.pow(2) - logvar_pred.exp(), )

        return 
    def loss_cls(self, pred_label,  mu_pred, logvar_pred ):

        loss_type ="sum"

        kld_pert = -0.5 * torch.sum(1. + logvar_pred - mu_pred.pow(2) - logvar_pred.exp(), )

        return 



    def train_epoch(self, train_loader,optimizer, epoch,share_encoder = True):
        self.model.train()
        total_loss = 0
        train_size = 0
        for drug_feautres,cell_features,label in tqdm(train_loader):
            # 假设 batch = (drug_fp, gene_idx, labels)
            
            drug_feautres = drug_feautres.to(self.dev)
            cell_features = cell_features.to(self.dev)
            label = label.to(self.dev)
            train_size += label.shape[0]
            optimizer.zero_grad()
            result, x2_pred , mu_pred, logvar_pred, z2_pred = self.model(cell_features, drug_feautres)  # forward
            assert not torch.isnan(x2_pred).any(), "x2_pred contains NaN"
            if self.model_type == 'reg' :
                loss,_1, _2, _3, _4, _5, _6  = self.loss_reg(result, mu_pred, logvar_pred)
            else :
                loss,_1, _2, _3, _4, _5, _6  = self.loss_cls(result, mu_pred, logvar_pred)
            
            # print(loss,_1,_2,_3,_4)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  

        return total_loss / train_size
    

    @torch.no_grad()
    def test(self, test_loader, loss_item, epoch, metrics_func=None,share_encoder=True):
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        
        self.model.eval()
        total_loss = 0
        test_size = 0
        all_scores, all_labels = [], []
        for drug_feautres,cell_features,label in tqdm(test_loader):
            # 假设 batch = (drug_fp, gene_idx, labels)
            drug_feautres = drug_feautres.to(self.dev)
            cell_features = cell_features.to(self.dev)
            label = label.to(self.dev)
            test_size += label.shape[0]

            result, x2_pred , mu_pred, logvar_pred, z2_pred = self.model(cell_features, drug_feautres) # forward

            loss_ls = self.loss_rec(result, mu_pred, logvar_pred)
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