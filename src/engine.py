import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
from utils import *


class GeneGraphEngine:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def loss(self, x1_train, x1_rec, x2_train, x2_rec, x2_pred, pred_adj, ori_adj ):
        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction="sum")
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction="sum")
        
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction="sum")

        # adj_loss = 
        return mse_x1 + mse_x2 +  mse_pert , \
                mse_x1, mse_x2, mse_pert 

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            # 假设 batch = (drug_fp, gene_idx, labels)
            drug_fp, gene_idx, labels = batch
            drug_fp = drug_fp.to(self.device)
            gene_idx = gene_idx.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            scores = self.model(drug_fp, gene_idx)  # forward
            loss = self.criterion(scores, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_loader)
    

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_scores, all_labels = [], []
        for batch in dataloader:
            drug_fp, gene_idx, labels = batch
            drug_fp = drug_fp.to(self.device)
            gene_idx = gene_idx.to(self.device)
            labels = labels.to(self.device).float()

            scores = self.model(drug_fp, gene_idx)
            loss = self.criterion(scores, labels)
            total_loss += loss.item()

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return total_loss / len(dataloader), all_scores, all_labels

    def fit(self, train_loader, val_loader, epochs=10):
        best_val_loss = float('inf')
        for epoch in range(1, epochs+1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_scores, val_labels = self.evaluate(val_loader)

            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")

    @torch.no_grad()
    def predict(self, drug_fp, gene_idx):
        self.model.eval()
        drug_fp = drug_fp.to(self.device)
        gene_idx = gene_idx.to(self.device)
        scores = self.model(drug_fp, gene_idx)
        return scores.cpu()