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

    def loss_VIB(self, x1_train, x1_rec, x2_train, x2_rec, x2_pred, pred_adj, ori_adj, x1_mu, x1_std, x2_mu, x2_std, mu, std ,epoch ):

        loss_type ="sum"
        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction=loss_type)
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction=loss_type)
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction=loss_type)
        x1_KL_loss = -0.5 * (1 + 2 * x1_std.log() - x1_mu.pow(2) - x1_std.pow(2)).sum(1).mean().div(math.log(2))
        x2_KL_loss = -0.5 * (1 + 2 * x2_std.log() - x2_mu.pow(2) - x2_std.pow(2)).sum(1).mean().div(math.log(2))
        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        return mse_x1 + mse_x2 +  mse_pert + x1_KL_loss+ x2_KL_loss + KL_loss,  \
                mse_x1, mse_x2, mse_pert , x1_KL_loss, x2_KL_loss,KL_loss



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