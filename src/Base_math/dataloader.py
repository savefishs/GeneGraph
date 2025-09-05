from doctest import Example
from locale import DAY_1
import numpy
from torch.utils.data import Dataset
import pickle
from .utils import *


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
        drug_path ="/root/myproject/GeneGraph/data/DrugComb/Process/reg/Drug_use.csv"
        cell_path = "/root/myproject/GeneGraph/data/DrugComb/Process/reg/Cell_use.csv"
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
    
    
    
