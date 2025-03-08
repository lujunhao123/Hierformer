# hier
import hl.utils as utils
import hl.TreeClass as treepkg
import hl.HierarchicalRegressor as regpkg
from hl.utils import adjust_learning_rate,string_split,EarlyStopping
from hl.result_print import All_Metrics,All_PI1_Metrics,All_PI2_Metrics


# normal
import scipy.cluster.hierarchy as sch
import scipy
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import platform
import time

# args
from parma_1 import parse_args,get_setting

# model | data
from model.must.Autoformer import Autoformer
from model.must.Crossformer import Crossformer
from model.must.Informer import Informer
from model.must.Timenet import TCN,LSTM,GRU
from model.must.Crossformer import Crossformer_Freq
from data_loader.data_factory_1 import _get_data
from model.quantile.quannet import htqf_loss,htqf_loss_hier
########################################################################################################################
import warnings
import random
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)

args = parse_args()
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print('Preprocessing \n')
system = platform.system()
args.data_split = string_split(args.data_split)
device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
model_dict = {
            'Autoformer': Autoformer,
            'Crossformer': Crossformer,
            'Informer': Informer,
            "GRU": GRU,
            'TCN': TCN,
            'LSTM': LSTM,
            'Crossformer_freq':Crossformer_Freq
        }

tau = np.r_[np.linspace(0.025, 0.075, 3), np.linspace(0.1, 0.9, 9), np.linspace(0.925, 0.975, 3)].reshape(1, -1)
tau_norm = scipy.stats.norm.ppf(tau)

params = {
        "d": 12, "num_layers": 6,"layer_dims": [120, 120, 60, 60, 10, 1],
        "nonlinear": "relu","lr": 1e-4,"num_epochs": 25,
        "quantiles": [0.025,0.050,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.925,0.950,0.975],
        "type": "median",
        "window": args.seq_len
    }

class _dict_parmsq():
    def __init__(self,params):
        self.__dict__.update(params)

params_q = _dict_parmsq(params)

########################################################################################################################
print(' \n Data PreProcessing phase \n')
print('Data Formatting \n')

site_files = [f for f in glob.glob('./io/input/' + '**_data.csv', recursive=True)] # './io/input\\Bear_preproc.csv'

if system=="Linux":
    sites = [site_file.split("/")[-1].split("_")[0] for site_file in site_files]  # linux系统 "\\"--"/"、【1】--【-1】
if system=="Windows":
    sites = [site_file.split("\\")[1].split("_")[0] for site_file in site_files]  # linux系统 "\\"--"/"、【1】--【-1】

computed_sites = []
sites = [site for site in sites if site not in computed_sites]

print('Data Integration \n')
#for site in sites:
site = args.site

if system=="Linux":
    site_file = [site_file for site_file in site_files if site_file.split('/')[-1].split('_')[0] == site][0] # linux系统 "\\"--"/"、【1】--【-1】
if system=="Windows":
    site_file = [site_file for site_file in site_files if site_file.split('\\')[1].split('_')[0] == site][0]  # linux系统 "\\"--"/"、【1】--【-1】


df = pd.read_csv(site_file, index_col=[0])
df.index = pd.to_datetime(df.index)
delta_t = pd.to_timedelta(df.index[1]-df.index[0])

wind_farms = df.columns.tolist()

########################################################################################################################
print('Formating \n')
########################################################################################################################
df_tree = dict()
add_weather_info = True
for wf in wind_farms:
    df_tree[wf] = df.loc[:, [wf]]
    df_tree[wf].rename(columns={wf: 'windpower'}, inplace=True)
# print(df_tree)

########################################################################################################################
print(' \n Mining: clustering')
########################################################################################################################
# 1. subset_of_wind_farms  = wind_farms # 可选择部分风电场
X = df.loc[:, wind_farms]   # X == df
cluster_thresholds = {'NSW1': 5500,"SA1":5500}   # NSW1:6  SA1:5
# 2. Hierarchical clustering
Z = sch.linkage(X.T.values, method='ward', optimal_ordering=False)
leaves_labels = np.array(X.columns)
# 3. Create S matrix from Z linkage matrix
S, y_ID = utils.create_Smatrix_from_linkagematrix(Z, leaves_labels)
threshold = cluster_thresholds[site]

########################################################################################################################
print(' \n Hierarchy initialization')
########################################################################################################################
### Creating spatial tree
S, y_ID = utils.cluster_Smatrix_from_threshold(Z, S, y_ID, threshold=threshold)   # 低于阈值的不要

tree_S = treepkg.Tree((S, y_ID), dimension='spatial')
# Creating data hierarchy
tree_S.create_spatial_hierarchy(df_tree, columns2aggr=['windpower'])
tree_H = tree_S
#print(tree_H.df)

extension = ''

########################################################################################################################
print(' \n Hierarchical Learning')
########################################################################################################################

# hierarchical forecasting methods:
#hierarchical_forecasting_methods = ['multitask', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']  # testing
hierarchical_forecasting_methods = ['OLS']
hierarchical_forecasting_method = 'BU' # 这个就是SG怎么出来，'multitask'、OLS、BU（这个是软约束用的方法）
# reconciliation methods: 'None', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV'
hierarchical_reconciliation_method = 'None' # 这个是后面做强制调整（可以理解为这个是强约束用的方法）

hreg = regpkg.H_Regressor(tree_H,forecasting_method=hierarchical_forecasting_method,
                          reconciliation_method=hierarchical_reconciliation_method,
                          alpha=0.75)


dfs = []
node_names = []
for node in tree_H.y_ID:
    df = tree_H.df[node]
    dfs.append(df)
    node_names.append(node)
result_df = pd.concat(dfs, axis=1, ignore_index=False)
result_df.columns = tree_H.y_ID
result_df.reset_index(inplace=True, drop=False)
print(result_df)
# result_df.to_csv("./io/input/hier_NSW1.csv")
# np.save("./io/input/NSW1_S.npy",tree_H.S)
# print(len(result_df.columns))  #NSW1:16-1=15 1:TIME
# G = np.zeros((tree_H.m, tree_H.n))
# for i, elem in enumerate(tree_H.leaves_label):
    #G[i, :] = [1 if id == elem else 0 for id in tree_H.y_ID]
    #SG = np.matmul(tree_H.S, G)
    #SG_T = torch.tensor(SG)
# np.save("./io/input/SA1_G.npy",G)
# print('SG',SG,'//',type(SG),'//',SG.shape) 140,140
# print('SG_T',SG_T,'//',type(SG_T),SG_T.shape) 140,140

hierarchical_forecasting_method = args.methods
print("hierarchical_forecasting_method:",hierarchical_forecasting_method)
hreg.hierarchical_forecasting_method = hierarchical_forecasting_method
# feature_num = len(result_df.columns) - 1
print('Args in experiment:')
print(args)
setting = get_setting(args)

train_dataset,train_loader = _get_data(args,"train",result_df)
val_dataset,val_loader = _get_data(args,"val",result_df)
test_dataset,test_loader = _get_data(args,"test",result_df)


model = model_dict[args.model]._Model_Q50(args).float()
model.train()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=args.patience, verbose=True)

if args.loss_fun=="mse":
    criterion = nn.MSELoss()
elif args.loss_fun=="htqf":
    criterion = htqf_loss(tau,tau_norm,args)
elif args.loss_fun=="htqf_hier":
    criterion = htqf_loss_hier(tau, tau_norm, args)

model_file_path = './checkpoint/' + setting + '/'
if not os.path.exists(model_file_path):
    os.makedirs(model_file_path)

model_filename = 'checkpoint.pth'
model_path = model_file_path+model_filename

def vail(vali_data, vali_loader, criterion):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for step, (seq_x, seq_x_y, seq_x_mark, seq_x_y_mark, seq_y, matrix_S,matrix_G,seq_xforq) in enumerate(vali_loader):
            mask = torch.zeros_like(seq_x_y)[:, -args.pred_len:, :].float().to(device)

            (seq_x, seq_x_y, seq_x_mark,
             seq_x_y_mark, seq_y, matrix_S, matrix_G, seq_xforq) = (seq_x.float().to(device), seq_x_y.float().to(device),
                                                                    seq_x_mark.float().to(device),
                                                                    seq_x_y_mark.float().to(device),
                                                                    seq_y.float().to(device),
                                                                    matrix_S.float().to(device),
                                                                    matrix_G.float().to(device),
                                                                    seq_xforq.float().to(device))
            # print(x.shape)
            dec_y = torch.cat([seq_x_y[:, :args.label_len, :], mask], dim=1).to(device)
            output = model(seq_x, seq_x_mark, dec_y, seq_x_y_mark, tau, tau_norm)
            true = seq_y
            if args.loss_fun=="mse":
                loss = criterion(output, true)
            elif args.loss_fun=="htqf":
                loss = criterion(output, true)
            elif args.loss_fun=="htqf_hier":
                loss = criterion(output, true,matrix_S[0],matrix_G[0],vali_data)
            test_loss.append(loss.item())

        total_loss = np.average(test_loss)
    model.train()
    return total_loss


for epoch in range(args.epochs):
    epoch_time = time.time()
    train_steps = len(train_loader)
    train_loss = []
    for step,(seq_x, seq_x_y, seq_x_mark, seq_x_y_mark, seq_y, matrix_S, matrix_G, seq_xforq) in enumerate(train_loader):
        mask = torch.zeros_like(seq_x_y)[:, -args.pred_len:, :].float().to(device)

        (seq_x, seq_x_y, seq_x_mark,
         seq_x_y_mark, seq_y, matrix_S, matrix_G, seq_xforq) = (seq_x.float().to(device), seq_x_y.float().to(device),
                                                                seq_x_mark.float().to(device),
                                                                seq_x_y_mark.float().to(device),
                                                                seq_y.float().to(device),
                                                                matrix_S.float().to(device),
                                                                matrix_G.float().to(device),
                                                                seq_xforq.float().to(device))
        # print(x.shape)
        dec_y = torch.cat([seq_x_y[:, :args.label_len,:], mask], dim=1).to(device)

        output = model(seq_x, seq_x_mark, dec_y, seq_x_y_mark, tau, tau_norm)
        true = seq_y

        if args.loss_fun == "mse":
            loss = criterion(output, true)
        elif args.loss_fun == "htqf":
            loss = criterion(output, true)
        elif args.loss_fun == "htqf_hier":
            loss = criterion(output, true, matrix_S[0], matrix_G[0], train_dataset)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    total_loss = np.average(train_loss)

    vali_loss = vail(val_dataset,val_loader,criterion)
    test_loss = vail(test_dataset,test_loader,criterion)

    print("Loss_fun：{0} Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
        args.loss_fun, epoch + 1, train_steps, total_loss, vali_loss, test_loss))

    early_stopping(vali_loss, model,model_path)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    #torch.save(model.state_dict(), model_path)
    adjust_learning_rate(optimizer, epoch + 1,args.lr)



def _predict(vali_data, vali_loader,model_file,flag):
    model = model_dict[args.model]._Model_Q50(args).float()
    model.to(device)
    setting = get_setting(args)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    hreg.coherency_constraint(y_diff=None)

    hierarchical_reconciliation_method = ["OLS", "BU", "STR", "hVAR", "sVAR", "COV", 'kCOV']
    for ii in range(len(hierarchical_reconciliation_method)):
        preds = []
        trues = []
        seq_xfor_q = []

        with torch.no_grad():
            for step, (seq_x, seq_x_y, seq_x_mark, seq_x_y_mark, seq_y, matrix_S,matrix_G,seq_xforq) in enumerate(vali_loader):
                mask = torch.zeros_like(seq_x_y)[:, -args.pred_len:, :].float().to(device)

                (seq_x, seq_x_y, seq_x_mark,
                 seq_x_y_mark, seq_y, matrix_S,matrix_G,seq_xforq) = (seq_x.float().to(device), seq_x_y.float().to(device),
                                                   seq_x_mark.float().to(device),
                                                   seq_x_y_mark.float().to(device), seq_y.float().to(device),
                                                   matrix_S.float().to(device),matrix_G.float().to(device),seq_xforq.float().to(device))
                # print(x.shape)
                dec_y = torch.cat([seq_x_y[:, :args.label_len, :], mask], dim=1).to(device)




                output = model(seq_x, seq_x_mark, dec_y, seq_x_y_mark, tau, tau_norm)


                # get SG
                if args.loss_fun=="mse":
                    true = seq_y
                    output_2d = output.contiguous().view(-1, output.size(-1))
                    output_2d_np = output_2d.cpu().detach().numpy()
                    true_2d = true.contiguous().contiguous().view(-1, true.size(-1))
                    true_2d_np = true_2d.cpu().detach().numpy()
                    y_diff = np.transpose(output_2d_np) - np.transpose(true_2d_np)

                    hreg.coherency_constraint(y_diff, method=hierarchical_reconciliation_method[ii])

                    SG = hreg.SG
                    SG_T = torch.tensor(SG)
                    SG_T = SG_T.float().to(device)

                    output_inversed = vali_data.inverse_transform_y(output)
                    output_must = torch.einsum('ij,bkj->bki', SG_T, output_inversed)
                    output = vali_data.transform_y(output_must)


                    preds.append(output.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                    seq_xfor_q.append(seq_xforq.detach().cpu().numpy())

                elif args.loss_fun=="htqf":
                    true = seq_y
                    output_forSG = output[...,7]
                    output_2d = output_forSG.contiguous().view(-1, output_forSG.size(-1))
                    output_2d_np = output_2d.cpu().detach().numpy()
                    true_2d = true.contiguous().contiguous().view(-1, true.size(-1))
                    true_2d_np = true_2d.cpu().detach().numpy()
                    y_diff = np.transpose(output_2d_np) - np.transpose(true_2d_np)

                    hreg.coherency_constraint(y_diff, method=hierarchical_reconciliation_method[ii])

                    SG = hreg.SG
                    SG_T = torch.tensor(SG)
                    SG_T = SG_T.float().to(device)
                    output_inversed = vali_data.inverse_transform_y(output)
                    output_inversed = output_inversed.float()
                    output_must = torch.einsum('ij,bkjf->bkif', SG_T, output_inversed)
                    output = vali_data.transform_y(output_must)

                    preds.append(output.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                    seq_xfor_q.append(seq_xforq.detach().cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            x_for_q = np.concatenate(seq_xfor_q,axis=0)

            if args.loss_fun=="mse":
                mae, rmse, mape, _ ,_ = All_Metrics(preds,trues)
                print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape * 100))
                folder_path = "./result_metrics/{}/{}+{}/".format(args.loss_fun,args.seq_len,args.pred_len)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = folder_path + '{}_{}.txt'.format(args.site,args.model)
                f = open(file_path, 'a')
                f.write("\n")
                f.write("========== {}_{}_{}_{} results ==========".format(args.model,hierarchical_reconciliation_method[ii],args.method,args.loss_fun))
                f.write("\n")
                f.write("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape * 100))
                f.write("\n")

            elif args.loss_fun=="htqf" or args.loss_fun=="htqf_hier":
                for tau_n in range(3):
                    picp, mpiw = All_PI2_Metrics(preds[..., tau_n], preds[..., -(tau_n + 1)], trues)
                    print("Average Horizon,PI {:.3f}, picp: {:.4f}%, mpiw: {:.2f}".format(
                        tau[0, -(tau_n + 1)] - tau[0, tau_n], picp * 100, mpiw))
                    crps, cross_loss = All_PI1_Metrics(preds, trues[:, :, :, None])
                    print("Average Horizon,crps {:.3f}, cross_loss: {:.2f}".format(
                        crps, cross_loss))

                    folder_path = "./result_metrics/{}/{}+{}/".format(args.loss_fun,args.seq_len,args.pred_len)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    file_path = folder_path + '{}_{}.txt'.format(args.site, args.model)
                    f = open(file_path, 'a')
                    f.write("\n")
                    f.write("========== {}_{}_{}_{} results ==========".format(args.model, args.hier_method, hierarchical_reconciliation_method[ii],
                                                                               args.loss_fun))
                    f.write("\n")
                    f.write("Average Horizon,PI {:.3f}, picp: {:.4f}%, mpiw: {:.2f}".format(
                        tau[0, -(tau_n + 1)] - tau[0, tau_n], picp * 100, mpiw))
                    f.write("\n")
                    f.write("Average Horizon,crps {:.3f}, cross_loss: {:.2f}".format(
                        crps, cross_loss))
                    f.write("\n")




            folder_path = './result/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if flag == "test":
                folder_path = './result/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                np.save(folder_path + "pred_Q50_{}.npy".format(hierarchical_reconciliation_method[ii]), preds)
                #np.save(folder_path + "true_Q50.npy",trues)
                # np.save(folder_path + "xforq_Q50.npy", x_for_q)

            elif flag == "train":
                folder_path = './result/' + setting + '/' + "train" + "/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                np.save(folder_path + "pred_Q50.npy", preds)
                np.save(folder_path + "true_Q50.npy", trues)
                np.save(folder_path + "xforq_Q50.npy", x_for_q)


_predict(test_dataset,test_loader,model_path,"test")








