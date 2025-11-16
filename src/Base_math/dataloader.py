from doctest import Example
from locale import DAY_1
from operator import index
import numpy
from torch.utils.data import Dataset
import pickle

from torch.utils.data.dataset import ConcatDataset
from .utils import *
import scipy.sparse as sp


## drop
class GeneGraphDataset(Dataset):

    def __init__(self, LINCS_index, mol_feature_type, mol_id, cid):

        self.LINCS_index = LINCS_index
        self.mol_feature_type = mol_feature_type
        self.mol_id = mol_id
        self.cid = cid
        # self.LINCS_data = load_from_HDF('../data/LINCS2020/processed_data.h5')
        self.LINCS_data = load_from_HDF('../data/LINCS2020/data_example/processed_data.h5')
        with open('../data/LINCS2020/idx2smi.pickle', 'rb') as f:
            self.idx2smi = pickle.load(f)
        if self.mol_feature_type == 'ECFP4':
            # with open('../data/LINCS2020/ECFP4_emb2048.pickle', 'rb') as f:
            with open('../data/LINCS2020/data_example/ECFP4_emb2048.pickle', 'rb') as f:
                self.smi2emb = pickle.load(f)
        elif self.mol_feature_type == 'KPGT':
            # with open('../data/LINCS2020/KPGT_emb2304.pickle', 'rb') as f:
            with open('../data/LINCS2020/data_example/KPGT_emb2304.pickle', 'rb') as f:
                self.smi2emb = pickle.load(f)
                

    def __getitem__(self, index):

        sub = subsetDict(self.LINCS_data, self.LINCS_index[index])
        mol_feature = self.smi2emb[self.idx2smi[self.mol_id[index]]]
        return sub['x1'], sub['x2'], mol_feature, self.mol_id[index], self.cid[index], sub['sig']

    def __len__(self):
        return self.mol_id.shape[0]


class GeneGraphDataset_screening(Dataset):

    def __init__(self, x1, mol_feature, mol_id, cid):

        self.x1 = x1
        self.mol_feature = mol_feature
        self.mol_id = mol_id
        self.cid = cid

    def __getitem__(self, index):

            return self.x1[index], self.mol_feature[index], self.mol_id[index], self.cid[index]

    def __len__(self):
        return self.mol_feature.shape[0]
    


class DrugCombDataset(Dataset):
    def __init__(self, csv_path, transforms=None,model_type = 'reg'):
        print(csv_path)
        drug_path ="../data/DrugComb/Process/reg/Drug_use.csv"
        cell_path = "../data/DrugComb/Process/reg/Cell_use.csv"
        data = pd.read_csv(csv_path)
        # self.example_list = pd.read_csv(csv_path)
        drug_list =  pd.read_csv(drug_path)
        cell_list = pd.read_csv(cell_path)
        
        self.drug_list =  drug_list.to_numpy()
        self.cell_list = cell_list.to_numpy()

        if model_type == 'reg':
            score_name = 'synergy_loewe'
        else :
            score_name = 'label'
        
        
        self.labels = data[score_name]
        # data = data.drop(columns=score_name)
        # data = data.drop(columns=score_name_drop)
        # data = data.drop(columns='Unnamed: 0')
        keep_cols = ["drug_row", "drug_col", "DepMap_ID"]
        data = data[keep_cols]
        data = np.array(data)
        self.inputs = data
        self.transforms = transforms

    def __getitem__(self, index):
        labels = self.labels[index]
        inputs_np = self.inputs[index]
        drug_features= self.drug_list[inputs_np[:-1]]
        print(drug_features)
        cell_features = self.drug_list[inputs_np[-1]]
        # inputs_tensor = torch.from_numpy(inputs_np).float()
        return drug_features,cell_features,labels
    
    def __len__(self):
        return len(self.labels.index)



## now using 
class OneDrugPretrain(Dataset):
    def __init__(self, csv_path,model_type):
        cell_base = '../data/dataprocess/M1_lincs_wth_ccle_org_all.csv'
        drug_cid = '../data/dataprocess/M1_lb_drug_to_smiles_cid.csv'
        drug_feature = '../data/dataprocess/L1000_ecfp4.npz'
        drug_pert_profile = '../data/dataprocess/pretrain_data_pert.csv'
        self.cell_list = pd.read_csv(cell_base)

        drug_list = pd.read_csv(drug_cid,sep='\t')
        feat_list = sp.load_npz(drug_feature)
        self.drug_pert = pd.read_csv(drug_pert_profile)
        drug_feat_list = pd.DataFrame(feat_list.toarray())
        self.drug_data = pd.concat([drug_list, drug_feat_list], axis=1)
        self.pert_to_feat = {
            row["pert_id"]: row[drug_feat_list.columns].values.astype(float)
            for _, row in self.drug_data.iterrows()
                }
        cell_set = set(self.cell_list['cell_iname'].unique())
        self.drug_pert = self.drug_pert[self.drug_pert['cell_mfc_name'].isin(cell_set)]
        if model_type == "pretrain" : 
            self.drug_pert = self.drug_pert[
                    (self.drug_pert["pert_idose"] == "10 uM") &
                    (self.drug_pert["pert_itime"] == "24 h")
                ].reset_index(drop=True)
        print(self.__len__())
    
    def __getitem__(self, index) :
        pert_row = self.drug_pert.iloc[index]
    
        pert_id = pert_row["pert_id"]
        cell_id  = pert_row['cell_mfc_name']
        # print(pert_id,cell_id)
        cell_row = self.cell_list[self.cell_list['cell_iname'] == cell_id].iloc[0]
        # 这里 cell_row 是一个 Series，可以选择返回整个 row 或者部分列
        # 比如只返回 gene expression 值：
        cell_expr = cell_row.drop(['Unnamed: 0','cell_iname']).values.astype(float)
        cell_expr = torch.tensor(cell_expr, dtype=torch.float32)

        pert_profile = pert_row.drop(["Unnamed: 0","sig_id", "pert_id", "cell_mfc_name", "pert_type", "pert_idose", "pert_itime"]).values.astype(float)
        mol_feat = torch.tensor(self.pert_to_feat[pert_id], dtype=torch.float32)

        pert_profile = torch.tensor(pert_profile,dtype =torch.float32)
        # print(cell_expr.shape)
        # print(cell_expr)
        return cell_expr, pert_profile, mol_feat
    
    def __len__(self):
        return len(self.drug_pert)
    

class DrugCombtraining(Dataset):
    def __init__(self,spilt ="train"):
        self.spilt = spilt

        cell_base = '../data/mydataset/M1_lincs_wth_ccle_org_all.csv'
        cell_turn = "../data/mydataset/M1_ccle_lincs_convert.csv"
        drug_cid = '../data/mydataset/M2_cid_smiles.csv'
        drug_feature = '../data/mydataset/M2_ecfp4.npz'
        drug_Comb = '../data/mydataset/M2_final_dataset.csv'
        drug_syn = '../data/mydataset/M2_SYN.pt'

        self.cell_list = pd.read_csv(cell_base)
        cell_turn = pd.read_csv(cell_turn)
        drug_list = pd.read_csv(drug_cid,sep=',')
        feat_list = sp.load_npz(drug_feature)
        drug_pert_full = pd.read_csv(drug_Comb,sep='\t')
        drug_feat_list = pd.DataFrame(feat_list.toarray())
        syn_ans_full = torch.load(drug_syn)




        mask = drug_pert_full.SPLIT == self.spilt
        self.drug_pert = drug_pert_full[mask]
        self.syn_ans = syn_ans_full[mask.values]  # 同步筛选 label

        split_cols = self.drug_pert['cid_cid_cell'].str.split('___', expand=True)
        # print(split_cols.head())
        self.drug_pert['pert_iname_A'] = split_cols.iloc[:, 0]
        self.drug_pert['pert_iname_B'] = split_cols.iloc[:, 1]
        self.drug_pert['cell_iname'] = split_cols.iloc[:, 2]

        self.drug_data = pd.concat([drug_list, drug_feat_list], axis=1)
        feat_cols = drug_feat_list.columns  # 0,1,2,...这些是 ECFP4 特征列

        # 用 CID 作为 key
        self.pert_to_feat = {
            str(row["CID"]): row[feat_cols].values.astype(float)
            for _, row in self.drug_data.iterrows()
        }
 
        cell_turn['ccle_name'] = cell_turn['ccle_name'].str.strip().str.upper()
        cell_turn['cell_iname'] = cell_turn['cell_iname'].str.strip().str.upper()
        self.cell_list['cell_iname'] = self.cell_list['cell_iname'].str.strip().str.upper()
        # 2. 建立映射字典
        ccle_to_cell = dict(zip(cell_turn['ccle_name'], cell_turn['cell_iname']))
        # 3. 用映射替换 drug_pert['cell_iname']
        self.drug_pert['cell_iname'] = self.drug_pert['cell_iname'].str.strip().str.upper().map(ccle_to_cell)


    def __len__(self):
        return len(self.drug_pert)
    
    def __getitem__(self, index):
        row = self.drug_pert.iloc[index]
        cell_name = row['cell_iname']
        # print(cell_name)
        # row_values = self.cell_list[self.cell_list.cell_iname==cell_name].iloc[0, 2:]
   
        cell_row = self.cell_list[self.cell_list['cell_iname'] == cell_name].iloc[0]
        cell_expr = cell_row.drop(['Unnamed: 0', 'cell_iname']).values.astype(float)
        cell_base = torch.tensor(cell_expr, dtype=torch.float32)

        pert_A = row['pert_iname_A']
        pert_B = row['pert_iname_B']
        features_A = torch.tensor(self.pert_to_feat[pert_A], dtype=torch.float32)
        features_B = torch.tensor(self.pert_to_feat[pert_B], dtype=torch.float32)

        label = torch.tensor(self.syn_ans[index], dtype=torch.float32)
        # print(cell_base.shape,label.shape,)
        return cell_base, features_A, features_B, label
        
