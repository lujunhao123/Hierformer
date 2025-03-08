import pandas as pd
import numpy as np
import time
import scipy.cluster.hierarchy as sch
import glob
import matplotlib.pyplot as plt

# Utility functions
import hl.utils as utils
from hl.utils import visual
import hl.TreeClass as treepkg
import hl.HierarchicalRegressor as regpkg

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.Former.models.Autoformer import Autoformer
from tqdm import tqdm
from statistics import mean



########################################################################################################################
print('Preprocessing \n')
########################################################################################################################
print(' \n Data PreProcessing phase \n')
print('Data Formatting \n')

site_files = [f for f in glob.glob('./io/input/' + '**_preproc.csv', recursive=True)] # './io/input\\Bear_preproc.csv'
sites = [site_file.split("\\")[1].split("_")[0] for site_file in site_files]
computed_sites = ['Bear', 'Bobcat', 'Bull', 'Cockatoo', 'Crow', 'Eagle', 'Fox', 'Gator', 'Hog', 'Lamb', 'Moose',
                  'Mouse', 'Panther', 'Peacock', 'Rat']
sites = [site for site in sites if site not in computed_sites]

print('Data Integration \n')
#for site in sites:
site = 'Fox'
site_file = [site_file for site_file in site_files if site_file.split('\\')[1].split('_')[0] == site][0]

df = pd.read_csv(site_file, index_col=[0])
df.index = pd.to_datetime(df.index)
delta_t = pd.to_timedelta(df.index[1]-df.index[0])

# 去掉不是建筑物的列
weather_cols = ['airTemperature', 'windSpeed', 'precipDepth1HR']
buildings = df.columns.tolist()
for col in weather_cols:
    buildings.remove(col)


# Drop columns with all NaNs
df.dropna(axis=1, how='all', inplace=True)   # 全是nan的删除
weather_cols_still_there = [col for col in df.columns.tolist() if col in weather_cols]

#print('df',df)
# All timestamp intervals
timestamp = pd.date_range(start=df.index[0], end=df.index[-1], freq='H')
df_newindex = pd.DataFrame(0, index=timestamp, columns=['todrop'])
df = pd.concat([df, df_newindex], axis=1)
df.drop('todrop', axis=1, inplace=True)
df.interpolate(method='time', inplace=True)
df.index.set_names('timestamp', inplace=True)

# 找一个阈值，处理缺失值
t_itv = utils.inner_intersect(df.reset_index(), col_ref=df.columns[0], col_loop=df.columns[1::])
# print(t_itv,'/n',t_itv['Consecutive_Measurements'],'/n',t_itv['BeginDate'],'/n',t_itv['EndDate'])
# Selecting the longest of them
idx_max = t_itv['Consecutive_Measurements'].astype(float).idxmax()
# Keep only largest consecutive time interval without NaNs
df_clean = df.loc[t_itv['BeginDate'][idx_max]: t_itv['EndDate'][idx_max]]
# print('df_clean',df_clean)
df_clean.interpolate(method='time', inplace=True)
df_clean.dropna(axis=1, inplace=True)

# 删除变化量为0占比超过30%列
variations = df_clean.diff()
pourcentage_of_null_variations = variations[variations == 0].count() / len(variations)
flat_columns = [col for col in variations.columns if pourcentage_of_null_variations[col] > 0.30]
df_clean.drop(flat_columns, axis=1, inplace=True)

df_clean.columns = df_clean.columns.tolist()
buildings = [col for col in df_clean.columns if col not in weather_cols]
########################################################################################################################
print('Formating \n')
########################################################################################################################
print('df_clean',df_clean)
df_tree = dict()
add_weather_info = True
for blg in buildings:

    if add_weather_info and len(weather_cols_still_there) > 0:
        df_tree[blg] = df_clean.loc[:, [blg]+weather_cols_still_there]
    else:
        df_tree[blg] = df_clean.loc[:, [blg]]

    df_tree[blg].rename(columns={blg: 'elec'}, inplace=True)
    add_weather_info = False

########################################################################################################################
print(' \n Mining: clustering')
########################################################################################################################

subset_of_buildings = buildings#[:20]
X = df_clean.loc[:, subset_of_buildings]   #X == df.clean
# print(df_clean.shape,X.shape)

cluster_thresholds = {'Fox': 40000}

### Hierarchical clustering
Z = sch.linkage(X.T.values, method='ward', optimal_ordering=False)
leaves_labels = np.array(X.columns)
# Create S matrix from Z linkage matrix
S, y_ID = utils.create_Smatrix_from_linkagematrix(Z, leaves_labels)
threshold = cluster_thresholds[site]
"""
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sch.dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()
"""

########################################################################################################################
print(' \n Hierarchy initialization')
########################################################################################################################
### Creating spatial tree
S, y_ID = utils.cluster_Smatrix_from_threshold(Z, S, y_ID, threshold=threshold)   # 低于阈值的不要
# 只到AA-AG因为就是不符合阈值,后面是叶的名字

tree_S = treepkg.Tree((S, y_ID), dimension='spatial')
# Creating data hierarchy
tree_S.create_spatial_hierarchy(df_tree, columns2aggr=['elec'])
tree_H = tree_S
# print(tree_H.df)
extension = ''

########################################################################################################################
print(' \n Hierarchical Learning')
########################################################################################################################

# hierarchical forecasting methods:
#hierarchical_forecasting_methods = ['multitask', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']  # testing
hierarchical_forecasting_methods = ['OLS']
hierarchical_forecasting_method = 'hVAR' # 这个就是SG怎么出来，'multitask'、OLS、BU（这个是软约束用的方法）
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
# print(len(result_df.columns))  141 去掉那个时间就140
# G = np.zeros((tree_H.m, tree_H.n))
# for i, elem in enumerate(tree_H.leaves_label):
    #G[i, :] = [1 if id == elem else 0 for id in tree_H.y_ID]
    #SG = np.matmul(tree_H.S, G)
    #SG_T = torch.tensor(SG)
# print('SG',SG,'//',type(SG),'//',SG.shape) 140,140
# print('SG_T',SG_T,'//',type(SG_T),SG_T.shape) 140,140
for hierarchical_forecasting_method in hierarchical_forecasting_methods:
    print("hierarchical_forecasting_method:",hierarchical_forecasting_method)
    hreg.hierarchical_forecasting_method = hierarchical_forecasting_method

    train_size = 0.75
    stand = StandardScaler()
    s_len = 64
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3
    epochs = 100
    pre_len = 32
    lab_len = 12
    from data_loader.dataloader import *

    train_x, test_x, train_y, test_y, scaler_x, scaler_y = read_data(result_df)
    # print('train_x', train_x[0])
    # q = scaler_x.transform(train_x[0][:,1:])
    # print('q',q)
    # f = scaler_x.inverse_transform(q)
    # print('f',f)

    train_data = Dataset(train_x,train_y,scaler_x, scaler_y)
    train_data = DataLoader(train_data, shuffle=True,  batch_size=batch_size)
    test_data = Dataset(test_x,test_y,scaler_x, scaler_y)
    test_data = DataLoader(test_data, shuffle=False,  batch_size=batch_size)

    model = Autoformer()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_train_loss_list = []
    epoch_test_loss_list = []

    for epoch in range(epochs):
        pbar = tqdm(train_data)
        for step,(x,y,xt,yt) in enumerate(pbar):
            preds = []
            trues = []
            train_loss = []

            if step == 0:
                hreg.coherency_constraint(y_diff=None)

            mask = torch.zeros_like(y)[:, -pre_len:, :].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :lab_len,:], mask], dim=1).to(device)

            output = model(x, xt, dec_y, yt)
            y = y[:, -pre_len:, :].to(device)
            predict = scaler_y.inverse_transform(output)
            predict_2d = predict.view(-1, predict.size(-1))
            predict_2d_np = predict_2d.cpu().detach().numpy()

            y_train_unscaled = scaler_y.inverse_transform(y)
            y_train_unscaled_2d = y_train_unscaled.view(-1, y_train_unscaled.size(-1))
            y_train_unscaled_2d_np = y_train_unscaled_2d.cpu().detach().numpy()

            y_diff = np.transpose(y_train_unscaled_2d_np)-np.transpose(predict_2d_np)

            criterion = CoherencyLossFunctionMSE(hreg) \
                if hreg.hierarchical_forecasting_method != 'multitask' \
                else IndependentLossFunctionMSE(hreg)

            loss = criterion(y, output, scaler_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hreg.coherency_constraint(y_diff)

            s = "[DOWN!]--train ==> Method: {} - Epoch: {} - Step: {} - Loss:{}".format(hreg.hierarchical_forecasting_method,epoch, step, loss.item())
            pbar.set_description(s)
            train_loss.append(loss.item())
        epoch_test_loss_list.append(mean(train_loss))

        model.eval()
        with torch.no_grad():
            pbar = tqdm(test_data)
            for step, (x, y, xt, yt) in enumerate(pbar):
                preds = []
                trues = []
                test_loss = []

                mask = torch.zeros_like(y)[:, -pre_len:, :].to(device)
                x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
                dec_y = torch.cat([y[:, :lab_len, :], mask], dim=1).to(device)

                output = model(x, xt, dec_y, yt)
                y = y[:, -pre_len:, :].to(device)

                loss = hreg.coherency_loss_function_mse(y, output, scaler_y) \
                    if hreg.hierarchical_forecasting_method != 'multitask' \
                    else hreg.independent_loss_function_mse(y, output)

                s = "[DOWN!]--test ==> Method: {} - Epoch: {} - Step: {} - Loss:{}".format(
                    hreg.hierarchical_forecasting_method, epoch, step, loss.item())
                pbar.set_description(s)
                test_loss.append(loss.item())

        epoch_test_loss_list.append(mean(test_loss))
        model.train()

    output_cpu = output.detach().cpu().numpy()
    y_cpu = y.detach().cpu().numpy()
    gt = output_cpu[0, :, -1]
    pd = y_cpu[0, :, -1]
    visual(gt, pd, './pic/' + 'Auto_test.svg')

    torch.save(model.state_dict(), './checkpoint/' + 'Auto_checkpoint.pth')


    plt.figure()
    plt.plot(epoch_train_loss_list, label='Auto_Training Loss', color='blue', linewidth=2)
    plt.plot(epoch_test_loss_list, label='Auto_Validation Loss', color='orange', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('./result/' + 'loss_Auto.svg',
                dpi=300, bbox_inches='tight', format="svg")
    plt.show()
    plt.close()






