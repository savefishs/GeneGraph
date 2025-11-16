from mimetypes import init
import sys
# from networkx import adj_matrix
from numpy import argsort
from Base_math.dataloader import OneDrugPretrain
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
    parser.add_argument("--save_path", type=str,default='../result/pretrain/')
    parser.add_argument("--molecule_path", type=str)
    parser.add_argument("--save_model",type=bool,default=False)
    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=364039)
    parser.add_argument("--molecule_feature", type=str, default='ECFP4', help='molecule_feature(KPGT, ECFP4)')
    parser.add_argument("--initialization_model", type=str, default='pretrain_shRNA', help='molecule_feature(pretrain_shRNA, random)')
    parser.add_argument("--split_data_type", type=str, default='smiles_split', help='split_data_type(random_split, smiles_split, cell_split)')
    parser.add_argument("--train_cell_count", type=str, default='None', help='if cell_split, train_cell_count=10,50,all, else None')

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--n_latent", type=int, default=100)
    parser.add_argument("--molecule_feature_embed_dim", nargs='+', type=int, default=[400])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--train_flag", type=bool, default=False)
    parser.add_argument("--eval_metric", type=bool, default=False)
    parser.add_argument("--predict_profile", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default='change_loss')
    
    args = parser.parse_args()
    return args

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file: str = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def train_genegraph(args):
    # config 



    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    data = load_from_HDF(args.data_path)


    #training  hyprparm
    subdir_name = f"{args.model_type}_bs{args.batch_size}_lr{str(args.learning_rate).replace('.', '_')}_ft{args.molecule_feature}"
    local_out =os.path.join("/root/myproject/GeneGraph/result/pretrain",subdir_name)
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


    mydataset=  OneDrugPretrain(csv_path=None, model_type="pretrain")

    n = len(mydataset)  # 数据集长度
    train_size = int(0.7 * n)
    val_size   = int(0.1 * n)
    test_size  = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(mydataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # model  
    ## init model
    if feat_type == "ECFP4" :
        features_dim = 2048
    

    model = TranSiGen(n_genes=978, n_latent=n_latent, n_en_hidden=[1200], n_de_hidden=[800],
                        features_dim=features_dim, features_embed_dim=features_embed_dim,
                        init_w=True, beta=beta, device=dev, dropout=dropout,
                        path_model=local_out, random_seed=random_seed)
    model_dict = model.state_dict()

    if init_mode == 'pretrain_shRNA':
        print('=====load vae for x1 and x2=======')
        filename = '/root/myproject/TranSiGen-main/results/trained_model_shRNA_vae_x1/best_model.pt'
        model_base_x1 = torch.load(filename, map_location='cpu')
        model_base_x1_dict = model_base_x1.state_dict()
        for k in model_dict.keys():
            if k in model_base_x1_dict.keys():
                model_dict[k] = model_base_x1_dict[k]
        filename = '/root/myproject/TranSiGen-main/results/trained_model_shRNA_vae_x2/best_model.pt'
        model_base_x2 = torch.load(filename, map_location='cpu')
        model_base_x2_dict = model_base_x2.state_dict()
        for k in model_dict.keys():
            if k in model_base_x2_dict.keys():
                model_dict[k] = model_base_x2_dict[k]
        model.load_state_dict(model_dict)
        del model_base_x1, model_base_x2
    model.to(dev)
    print(model)

    engine_model = GeneGraphEngine(model=model,beta=args.beta,device=dev)

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
        train_dict, train_metrics_dict, train_metricsdict_ls =engine_model.test(test_loader = train_loader, loss_item = loss_item)
        valid_dict, valid_metrics_dict, valid_metricsdict_ls =engine_model.test(test_loader = val_loader, loss_item = loss_item)
       
        
         
        valid_loss = valid_dict['loss']
        
        train_str = ", ".join([f"{k}={v:.4f}" for k, v in train_dict.items()])
       
        valid_str = ", ".join([f"{k}={v:.4f}" for k, v in valid_dict.items()])
        
        r_str = f"Epoch {epoch+1}/{n_epochs}: Train: {train_str}   Valid: {valid_str}"
        print(r_str)

        

        with open(log_file_path, "a") as f:
                f.write(r_str + "\n")

        if valid_loss < best_value:
            best_value = valid_loss
            best_epoch = epoch
            print("save best model at epoch : {}",best_epoch)
            if save_model:
                torch.save(model, os.path.join(local_out, "best_model.pt"))
    

    ## evaling
    test_dict, test_metrics_dict, test_metricsdict_ls =engine_model.test(test_loader = test_loader, loss_item = loss_item)
    test_str = ", ".join([f"{k}={v:.4f}" for k, v in test_dict.items()])
    test_str = "Best model :" + test_str
    print(test_str)
    with open(log_file_path, "a") as f:
        f.write(test_str + "\n")


if __name__ == "__main__" : 
    args = parse_args()
    train_genegraph(args)
