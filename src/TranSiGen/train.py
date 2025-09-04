from mimetypes import init
# from networkx import adj_matrix
from numpy import argsort
from Base_math.dataloader import DrugCombDataset
import scipy.sparse as sp
from .model_TranSiGen import TranSiGen 
from .TranSiGen_engine import TranSiGen_engine

from Base_math.utils import *
import logging
import pickle
import argparse
import json
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training Gene Graph")
    parser.add_argument("--data_path", type=str,default='../data/LINCS2020/data_example/processed_data_id.h5')
    parser.add_argument("--save_path", type=str,default='../result/test/')
    parser.add_argument("--molecule_path", type=str)
    parser.add_argument("--save_model",action='store_true')
    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=364039)
    parser.add_argument("--molecule_feature", type=str, default='KPGT', help='molecule_feature(KPGT, ECFP4)')
    parser.add_argument("--initialization_model", type=str, default='pretrain_shRNA', help='molecule_feature(pretrain_shRNA, random)')
    parser.add_argument("--split_data_type", type=str, default='smiles_split', help='split_data_type(random_split, smiles_split, cell_split)')
    parser.add_argument("--train_cell_count", type=str, default='None', help='if cell_split, train_cell_count=10,50,all, else None')

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--n_latent", type=int, default=100)
    parser.add_argument("--molecule_feature_embed_dim", nargs='+', type=int, default=[400])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--train_flag", type=bool, default=False)
    parser.add_argument("--eval_metric", type=bool, default=False)
    parser.add_argument("--predict_profile", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default='Combine')
    args = parser.parse_args()
    return args

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file: str = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

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



def train_genegraph(args,repeat,fold):
    # config 

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    data_path = f'/root/myproject/GeneGraph/data/DrugComb/Process/reg/data'



    # hyprparm
    local_out =f"/root/myproject/GeneGraph/result/{args.model_type}/Combine_repeat{repeat}_fold{fold}"
    n_epochs = args.n_epochs
    n_latent = args.n_latent
    split_type = args.split_data_type
    n_folds = 5
    train_cell_count = args.train_cell_count
    feat_type = args.molecule_feature
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    beta = args.beta
    dropout = args.dropout
    weight_decay = args.weight_decay
    save_model = args.save_model

    args.init_w=True
    args.beta=beta 
    args.device=dev
    args.dropout=dropout
                            
    args.path_model=local_out 
    args.random_seed=random_seed
    args.graph_skip_conn = 0.7
    args.graph_include_self = True
    # data 



    train = DrugCombDataset( os.path.join(data_path,f"repeat{repeat}_fold{fold}_train.csv"))

    valid = DrugCombDataset( os.path.join(data_path,f"repeat{repeat}_fold{fold}_val.csv"))

    test = DrugCombDataset( os.path.join(data_path,f"repeat{repeat}_fold{fold}_test.csv"))

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)

    # model  
    ## init model


    model = TranSiGen(n_genes=978, n_latent=n_latent, n_en_hidden=[1200], n_de_hidden=[800],
                                    features_dim=2048, features_embed_dim=[400],
                                    init_w=True, beta=beta, device=dev, dropout=dropout,
                                    path_model=local_out, random_seed=random_seed)
    
    old_model = torch.load('best_model.pt')
    old_state = old_model.state_dict()
    model.load_state_dict(old_state, strict=False)
    ## load_state


    engine_model = TranSiGen_engine(model=model,device=dev)

    ## trainning
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam( model.parameters(),lr = learning_rate,
        weight_decay=weight_decay
        )
    loss_item = ['loss', 'mse_x1', 'mse_x2', 'mse_pert',"adj_loss"]

    best_value = np.inf
    best_epoch = 0
    for epoch in range(n_epochs):
        log = engine_model.train_epoch(train_loader, optimizer,epoch)
        train_dict, train_metrics_dict, train_metricsdict_ls =engine_model.test(train_loader, , loss_item, epoch)
        test_dict, test_metrics_dict, test_metricsdict_ls =engine_model.test(valid_loader, , loss_item, epoch)
       
        
         
        test_loss = test_dict['loss']
        
    
        r_str = f"Epoch {epoch+1}/{n_epochs}: Train: loss={train_dict['loss']:.4f}, mse_x1={train_dict['mse_x1']:.4f}, mse_x2={train_dict['mse_x2']:.4f}, mse_pert={train_dict['mse_pert']:.4f}, adj_loss={train_dict['adj_loss']:.4f}   Valid: loss={test_dict['loss']:.4f}, mse_x1={test_dict['mse_x1']:.4f}, mse_x2={test_dict['mse_x2']:.4f}, mse_pert={test_dict['mse_pert']:.4f}, adj_loss={test_dict['adj_loss']:.4f} "

        print(r_str)
        log_file_path = os.path.join(local_out,"train_log.txt")

        with open(log_file_path, "a") as f:
                f.write(r_str + "\n")

        if test_loss < best_value:
            best_value = test_loss
            best_epoch = epoch
            if save_model:
                torch.save(model, os.path.join(local_out, "best_model.pt"))
    

    ## evaling
    test_dict, test_metrics_dict, test_metricsdict_ls =engine_model.test(test_loader, adj_torch, loss_item,epoch=500)

    


if __name__ == "__main__" : 
    args = parse_args()
    for repeat in range(1,11):
        for fold in range(1,6):
            train_genegraph(args,repeat=repeat,fold=fold)
