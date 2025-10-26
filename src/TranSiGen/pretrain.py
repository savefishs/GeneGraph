from mimetypes import init
import sys
# from networkx import adj_matrix
from numpy import argsort
from Base_math.dataloader import DrugCombtraining
import scipy.sparse as sp
from .model import TranSiGen
from .pretrain_engine import GeneGraphEngine
from Base_math.utils import *
from torch.utils.data import random_split
import TranSiGen_onepre.vae_x1  as vae_x1
sys.modules["vae_x1"] = vae_x1 
import TranSiGen_onepre.vae_x2  as vae_x2
sys.modules["vae_x2"] = vae_x2
import logging
import pickle
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training Gene Graph")
    parser.add_argument("--data_path", type=str,default='../data/LINCS2020/data_example/processed_data_id.h5')
    parser.add_argument("--save_path", type=str,default='../result/Combine/')
    parser.add_argument("--molecule_path", type=str)
    parser.add_argument("--save_model",type=bool,default=True)
    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=364039)
    parser.add_argument("--molecule_feature", type=str, default='ECFP4', help='molecule_feature(KPGT, ECFP4)')
    parser.add_argument("--initialization_model", type=str, default='pretrain_shRNA', help='molecule_feature(pretrain_shRNA, random)')
    parser.add_argument("--split_data_type", type=str, default='smiles_split', help='split_data_type(random_split, smiles_split, cell_split)')
    parser.add_argument("--train_cell_count", type=str, default='None', help='if cell_split, train_cell_count=10,50,all, else None')

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--n_latent", type=int, default=100)
    parser.add_argument("--molecule_feature_embed_dim", nargs='+', type=int, default=[400])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--train_flag", type=bool, default=False)
    parser.add_argument("--eval_metric", type=bool, default=False)
    parser.add_argument("--predict_profile", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default='norm')
    
    args = parser.parse_args()
    return args

def train_genegraph(args):
    # config 

    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')

    #training  hyprparm
    subdir_name = f"{args.model_type}_bs{args.batch_size}_lr{str(args.learning_rate).replace('.', '_')}_ft{args.molecule_feature}"
    local_out =os.path.join("/root/myproject/GeneGraph/result/Combine",subdir_name)
    if not os.path.exists(local_out):
        os.makedirs(local_out)
    log_file_path = os.path.join(local_out, 'train_log.txt')

    argparse_dict = vars(args)
    with open(os.path.join(local_out, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    n_epochs = args.n_epochs
    n_latent = args.n_latent
    split_type = args.split_data_type
    n_folds = 5
    train_cell_count = args.train_cell_count
    
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    beta = args.beta
    dropout = args.dropout
    weight_decay = args.weight_decay
    save_model = args.save_model
    init_mode = args.initialization_model

    # model parms 
    feat_type = args.molecule_feature
    features_embed_dim = args.molecule_feature_embed_dim
    

    args.init_w=True
    args.beta=beta 
    args.device=dev
    args.dropout=dropout
                            
    args.path_model=local_out 
    args.random_seed=random_seed
    args.graph_skip_conn = None
    args.graph_include_self = True
    # data 


    train_dataset = DrugCombtraining(spilt = "train")
    val_dataset = DrugCombtraining(spilt = 'val')
    test_dataset = DrugCombtraining(spilt = 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # model  
    ## init model
    if feat_type == "ECFP4" :
        features_dim = 2048
    

    model = TranSiGen(n_genes=978, n_latent=n_latent, n_en_hidden=[1200], n_de_hidden=[800],
                        features_dim=features_dim, features_embed_dim=features_embed_dim,
                        init_w=True, beta=beta, device=dev, dropout=dropout,
                        path_model=local_out, random_seed=random_seed)
    model_dict = model.state_dict()

    pretrained_path = '/root/myproject/GeneGraph/result/pretrain/norm_bs64_lr0_001_ftECFP4/best_model.pt'
    pretrained_model = torch.load(pretrained_path, map_location='cpu')
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    # 只加载 VAE 相关模块
    for k in model_dict.keys():
        if k in pretrained_dict and ('encoder_x1' in k or 'decoder_x1' in k or 
                                    'mu_z1' in k or 'logvar_z1' in k or 
                                    'encoder_x2' in k or 'decoder_x2' in k or 
                                    'mu_z2' in k or 'logvar_z2' in k):
            model_dict[k] = pretrained_dict[k]

    model.load_state_dict(model_dict)
    del pretrained_model

    # -------------------------
    # 3. 冻结 VAE 模块，只训练分类器和 cross-attention
    # -------------------------
    for name, param in model.named_parameters():
        if 'classifier' in name or 'cross_attention' in name or "proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    engine_model = GeneGraphEngine(model=model,beta=args.beta,device=dev)

    for name, param in model.named_parameters():
        print(f"{name:50s} | requires_grad={param.requires_grad}")
    print("========================================")
    ## trainning
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam( model.parameters(),lr = learning_rate,
        weight_decay=weight_decay
        )
    
    loss_item = ['loss', 'mse_x1', 'mse_x2', 'mse_pert',"kld_x1", "kld_x2", "kld_pert"]
    # loss_item = ['loss', 'mse_x1', 'mse_x2', 'KL1',"KL2"]

    best_value = np.inf
    best_epoch = 0
    for epoch in range(n_epochs):
        log = engine_model.train_epoch(train_loader, optimizer)
        # train_mse,train_pcc =engine_model.test(test_loader = train_loader, loss_item = loss_item)
        valid_mse,valid_pcc =engine_model.test(test_loader = val_loader, loss_item = loss_item)
       
        r_str = (f"Epoch {epoch+1}/{n_epochs}: "
                f"Train: MSE={log:.4f}, |"
                f"Valid: MSE={valid_mse:.4f}, PCC={valid_pcc:.4f}")
        print(r_str)

    

        with open(log_file_path, "a") as f:
                f.write(r_str + "\n")

        if valid_mse < best_value:
            best_value = valid_mse
            best_epoch = epoch
            print("save best model at epoch : {}",best_epoch)
            if save_model:
                torch.save(model, os.path.join(local_out, "best_model.pt"))
    
    ## evaling
    test_mse,test_pcc =engine_model.test(test_loader = test_loader, loss_item = loss_item)
    test_str = (f"Valid: MSE={test_mse:.4f}, PCC={test_pcc:.4f}")
    test_str = "Best model :" + test_str
    print(test_str)


if __name__ == "__main__" : 
    args = parse_args()
    train_genegraph(args)
