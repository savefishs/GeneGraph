from collections import OrderedDict
import os
import torch
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp
import torch.nn.functional as F
import random
import h5py
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader

def setup_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_from_HDF(fname):
    """Load data from a HDF5 file to a dictionary."""
    data = dict()
    with h5py.File(fname, 'r') as f:
        for key in f:
            data[key] = np.asarray(f[key])
            if isinstance(data[key][0], np.bytes_):
                data[key] = data[key].astype(np.str)
    return data

def save_to_HDF(fname, data):
    """Save data (a dictionary) to a HDF5 file."""
    with h5py.File(fname, 'w') as f:
        for key, item in data.items():
            if isinstance(item[0], np.str):
                item = item.astype(np.bytes_)
            f[key] = item

def selectFromDict(d, keys):
    '''select the chosen @keys in dict @d'''
    res = dict()
    for k in keys:
        if k not in d.keys():
            raise ValueError("Key " + str(k) + " not found")
        else:
            res[k] = d[k]
    return res

def subsetDict(d, ind):
    '''subset each numpy array in dict @d to indices @ind'''
    res = dict()
    if not isinstance(ind, np.ndarray):
        ind = np.asarray(ind)
    for k in d.keys():
        if isinstance(d[k], np.ndarray):
            res[k] = d[k][ind]
    return res

def concatDictElemetwise(a, b):
    '''concatenate dict @a and @b elementwise'''
    res = dict()
    if not sorted(a.keys()) == sorted(b.keys()):
        raise ValueError("Mismatching key sets")
    for k in a.keys():
        res[k] = np.concatenate((a[k], b[k]), axis=0)
    return res

def concat_dicts(a, b):
    # works for both dict and OrderedDict classes
    assert (len(set(a.keys()) & set(b.keys())) == 0), "Can't unambiguously concatenate."
    return type(a)(list(a.items()) + list(b.items()))

def getSplitsByGroupKFold(groups, n_splits, shuffle, random_state):
    assert (n_splits >= 3)
    kf = GroupKFold(n_splits=n_splits)

    if shuffle:
        # randomly rename groups so that the GroupKFold (which sorts by group ids first) splits can be randomized
        unique_groups = np.unique(groups)
        rnd_renames = sklearn.utils.shuffle(np.arange(len(unique_groups)), random_state=random_state)
        groups_renamed = np.array([rnd_renames[np.argwhere(unique_groups == g)[0]] for g in groups])
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups_renamed)
    else:
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups)

    folds = [list(x[1]) for x in kfsplit]
    folds_nums = list(range(len(folds)))

    tr_fold_nums = folds_nums[:-2]
    ind_tr = sum([folds[i] for i in tr_fold_nums], [])
    ind_va = folds[folds_nums[-2]]
    ind_te = folds[folds_nums[-1]]

    return ind_tr, ind_va, ind_te


def split_data(data, n_folds=5, split_type='random_split', rnds=None):
    if split_type == 'random_split':
        ind_tr, ind_va, ind_te = getSplitsByGroupKFold(data['sig'], n_folds, shuffle=True, random_state=rnds)
    elif split_type == 'smiles_split':
        ind_tr, ind_va, ind_te = getSplitsByGroupKFold(data['canonical_smiles'], n_folds, shuffle=True, random_state=rnds)
    ppair = subsetDict(data, ind_tr)
    ppairv = subsetDict(data, ind_va)
    ppairt = subsetDict(data, ind_te)

    return ppair, ppairv, ppairt

def split_data_cid(data, train_cell_count='all'):
    cell_ls_pretrain = ['HT29', 'NPC', 'A375', 'PC3', 'ASC', 'MCF7', 'HCC515', 'VCAP', 'HA1E', 'A549']

    df_data = pd.DataFrame(data['cid'], columns=['cid'])
    df_data_cid = df_data.value_counts('cid').reset_index()
    df_data_cid.columns = ['cid', 'count']

    test_cell_ls = df_data_cid[~df_data_cid['cid'].isin(cell_ls_pretrain)].sample(n=7, replace=False, random_state=1234)['cid'].tolist()
    valid_cell_ls = df_data_cid[~df_data_cid['cid'].isin(cell_ls_pretrain + test_cell_ls)].sample(n=7, replace=False, random_state=1234)['cid'].tolist()

    if train_cell_count == 'all':
        train_cell_ls = df_data_cid[~(df_data_cid['cid'].isin(valid_cell_ls + test_cell_ls))]['cid'].tolist()
    else:
        train_cell_ls = df_data_cid[~df_data_cid['cid'].isin(valid_cell_ls + test_cell_ls)].sample(n=int(train_cell_count), replace=False, random_state=1234)['cid'].tolist()

    index_ls = df_data[df_data['cid'].isin(train_cell_ls)].index.tolist()
    ppair = subsetDict(data, index_ls)
    index_ls = df_data[df_data['cid'].isin(valid_cell_ls)].index.tolist()
    ppairv = subsetDict(data, index_ls)
    index_ls = df_data[df_data['cid'].isin(test_cell_ls)].index.tolist()
    ppairt = subsetDict(data, index_ls)
    return ppair, ppairv, ppairt


def precision_10(label_test, label_predict):
    k = 10
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k

def precision_20(label_test, label_predict):
    k = 20
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k

def precision_50(label_test, label_predict):
    k = 50
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k

def precision_100(label_test, label_predict):
    k = 100
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k



def rmse(targets, preds):

    return np.sqrt(mean_squared_error(targets, preds))


def pearson(targets, preds):
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    try:
        return pearsonr(targets, preds)[0]
    except ValueError:
        print(targets, preds)
        print(np.isnan(targets), np.isnan(preds))
        return float('nan')


def spearman(targets, preds):
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    try:
        return spearmanr(targets, preds)[0]
    except ValueError:
        return float('nan')


# This file consists of useful functions that are related to cmap
def computecs(qup, qdown, expression):
    '''
    This function takes qup & qdown, which are lists of gene
    names, and  expression, a panda data frame of the expressions
    of genes as input, and output the connectivity score vector
    '''
    r1 = ranklist(expression)
    if qup and qdown:
        esup = computees(qup, r1)
        esdown = computees(qdown, r1)
        w = []
        for i in range(len(esup)):
            if esup[i]*esdown[i] <= 0:
                w.append(esup[i]-esdown[i])
            else:
                w.append(0)
        return pd.DataFrame(w, expression.columns)
    elif qup and qdown==None:
        print('None down')
        esup = computees(qup, r1)
        return pd.DataFrame(esup, expression.columns)
    elif qup == None and qdown:
        print('None up')
        esdown = computees(qdown, r1)
        return pd.DataFrame(esdown, expression.columns)
    else:
        return None

def computees(q, r1):
    '''
    This function takes q, a list of gene names, and r1, a panda data
    frame as the input, and output the enrichment score vector
    '''
    if len(q) == 0:
        ks = 0
    elif len(q) == 1:
        ks = r1.loc[q, :]
        ks.index = [0]
        ks = ks.T

    else:
        n = r1.shape[0]

        sub = r1.loc[q, :]
        J = sub.rank()

        a_vect = J/len(q)-sub/n
        b_vect = sub / n - (J - 1) / len(q)
        a = a_vect.max()
        b = b_vect.max()

        ks = []
        for i in range(len(a)):
            if a[i] > b[i]:
                ks.append(a[i])
            else:
                ks.append(-b[i])

    return ks

def ranklist(DT):
    # This function takes a panda data frame of gene names and expressions
    # as an input, and output a data frame of gene names and ranks
    ranks = DT.rank(ascending=False, method="first")
    return ranks

def get_metric_func(metric):
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'pearson':
        return pearson

    if metric == 'spearman':
        return spearman

    if metric == 'precision10':
        return precision_10

    if metric == 'precision20':
        return precision_20

    if metric == 'precision50':
        return precision_50

    if metric == 'precision100':
        return precision_100

    raise ValueError(f'Metric "{metric}" not supported.')


def calc_MODZ(array):

    array = array.data.cpu().numpy().astype(float)

    if len(array) == 2:
        return torch.Tensor([np.mean(array, 0)])
    else:
        CM = spearmanr(array.T)[0]

        weights = np.sum(CM, 1) - 1
        weights = weights / np.sum(weights)
        weights = weights.reshape((-1, 1))
        return torch.Tensor([np.dot(array.T, weights).reshape((-1, 1)[0])])
    

def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

## add

def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


def normalize_adj_soft(adj):
    adj = adj + 1e-8  # avoid log(0)
    return adj / adj.sum(dim=(-2, -1), keepdim=True)


def batched_kl_div(P, Q):
    """
    P, Q: [B, N, N], must be normalized row-wise or via softmax
    Returns average KL divergence over batch
    """
    P_log = torch.log(P + 1e-8)  # avoid log(0), shape: [B, N, N]
    kl = F.kl_div(P_log, Q, reduction='none')  # keep full shape
    return kl.sum(dim=(-1, -2)).mean()         # sum over N×N, then mean over batch

def laplacian_smoothness_loss(A: torch.Tensor, X: torch.Tensor):
    """
    Batch version using vectorized operations.
    """
    D = torch.diag_embed(A.sum(dim=-1))     # [B, N, N]
    L = D - A                                # [B, N, N]
    X_t = X.transpose(1, 2)                  # [B, d, N]
    LX = torch.bmm(L, X)                     # [B, N, d]
    XLX = torch.bmm(X_t, LX)                 # [B, d, d]
    trace_per_batch = XLX.diagonal(offset=0, dim1=1, dim2=2).sum(-1)  # [B]
    return trace_per_batch.mean()




def split(data, repeat):
    kf = KFold(n_splits=5, random_state=repeat, shuffle=True)
    seed = 42
    i = 1
    for train_index , test_index in kf.split(data):
        data_train_val = data.iloc[train_index]
        data_test = data.iloc[test_index]
        data_train, data_val = train_test_split(data_train_val, test_size=0.25, random_state=seed)
        all_index = np.concatenate((data_train.index.values, data_val.index.values, data_test.index.values), axis=0)
        all_index = np.unique(all_index)

        data_train = data_train.reset_index(drop=True)
        data_val = data_val.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)
        path_train = './data/repeat'+str(repeat)+'_fold'+str(i)+'_train.csv'
        path_val = './data/repeat'+str(repeat)+'_fold'+str(i)+'_val.csv'
        path_test = './data/repeat'+str(repeat)+'_fold'+str(i)+'_test.csv'
        i += 1
        print(path_train)
        data_train.to_csv(path_train)
        data_val.to_csv(path_val)
        data_test.to_csv(path_test)


def format_metrics(epoch, n_epochs, train_dict, test_dict):
    def format_dict(d):
        parts = []
        for k, v in d.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    train_str = format_dict(train_dict)
    test_str = format_dict(test_dict)

    return f"Epoch {epoch+1}/{n_epochs}: Train: {train_str}   Valid: {test_str}"