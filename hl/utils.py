import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
device = "cuda" if torch.cuda.is_available() else "cpu"
#######################tool###########################################
def flatten(list_of_lists):
    """"Source: https://stackabuse.com/python-how-to-flatten-list-of-lists/"""
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

################################data-processing##########################
def consecutive_measurements(df_in, col_object, col_index='timestamp'):
    """Function to count consecutive measurements and missing values within a dataframe column.
    Adapted from: https://stackoverflow.com/questions/26911851/how-to-use-pandas-to-find-consecutive-same-data-in-time-series.

    Parameters:
    df_in:      dataframe object
    col_object: string object pointing to the column of interest
    col_index:  string object pointing to a datatime column (typially a dataframe index set as a column)"""

    df_manip = df_in.copy()
    # Counting NaN values as int where 1 = NaN, and 0 = measurement
    df_manip["Nan_int"] = df_manip[col_object].isna().astype('int')

    # Obtaining the cumulative counts of values
    df_manip['value_grp'] = (df_manip["Nan_int"].diff(1) != 0).astype('int').cumsum()
    # Grouping them into a frame and extraacting begin and end Dates of cumulative sequences
    df_nanfilter = pd.DataFrame({'BeginDate': df_manip.groupby('value_grp')[col_index].first(),
                                 'EndDate': df_manip.groupby('value_grp')[col_index].last(),
                                 'Consecutive': df_manip.groupby('value_grp').size(),
                                 'Val': df_manip.groupby('value_grp').Nan_int.first()})
    return df_nanfilter

def consecutive_missingval_thershold(df_nanfilter,
                                     threshold: int = 4):
    """Adding a threshold for consecutive missing value consideration"""

    df_nanfilter_collapsed = df_nanfilter.copy()
    # Looping over the missing values below the given threshold
    for index, row in df_nanfilter.iterrows():

        # Skip first and last row if missing vals are obeserved
        if index == 1 and row["Val"] == 1:
            continue
        elif index == len(df_nanfilter) and row["Val"] == 1:
            continue

        # Otherwise proceed to collapse the dataframe
        elif row["Val"] == 1 and row["Consecutive"] < threshold:
            identifying_moving_index = df_nanfilter_collapsed["EndDate"] == df_nanfilter.loc[index-1, "EndDate"]
            index_in_collapsing_df = list(df_nanfilter_collapsed[identifying_moving_index].index)[0]

            # Extend the date of the measured data - index-1 row
            df_nanfilter_collapsed.loc[index_in_collapsing_df, "EndDate"] = df_nanfilter.loc[index + 1, "EndDate"]
            # Sum the consecutive values of missing & measured data
            sum_consecutive = df_nanfilter.loc[index - 1, "Consecutive"] + row.loc["Consecutive"] + df_nanfilter.loc[
                index + 1, "Consecutive"]
            df_nanfilter.loc[index - 1, "Consecutive"] = sum_consecutive
            # Collapse the frame
            df_nanfilter_collapsed.drop([index_in_collapsing_df + 1, index_in_collapsing_df + 2], inplace=True)
            df_nanfilter_collapsed.reset_index(drop=True, inplace=True)

    return df_nanfilter_collapsed


def inner_intersect(df_in,
                    col_ref: str,
                    col_loop: list):
    """This function returns the inner intersection of consecutive measurements between columns of an input dataframe"""

    df_nanfilter = consecutive_measurements(df_in, col_object=col_ref, col_index='timestamp')
    # print(df_nanfilter)
    df_nanfilter = consecutive_missingval_thershold(df_nanfilter, threshold=8)
    df_measurements1 = df_nanfilter[df_nanfilter["Val"] == 0]
    # print('df_nanfilter',df_nanfilter)
    t_itv = pd.DataFrame(columns=["BeginDate", "EndDate", "Consecutive_Measurements"])
    for index1, row1 in df_measurements1.iterrows():
        empty_intersect = False
        t_start_max, t_end_min = row1["BeginDate"], row1["EndDate"]
        # print('Main Loop Row1 : ' +str(row1["BeginDate"]))

        for col in col_loop:
            df_nanfilter2 = consecutive_measurements(df_in, col_object=col, col_index='timestamp')
            df_nanfilter2 = consecutive_missingval_thershold(df_nanfilter2, threshold=4)
            df_measurements2 = df_nanfilter2[df_nanfilter2["Val"] == 0]

            for index2, row2 in df_measurements2.iterrows():
                t_start = max(row1["BeginDate"], row2["BeginDate"])
                t_end = min(row1["EndDate"], row2["EndDate"])

                if t_start > t_end:  # no intersect
                    empty_intersect = True
                    pass
                else:
                    empty_intersect = False
                    t_start_max = max(t_start, t_start_max)
                    t_end_min = max(t_end, t_end_min)
                    pass

                if empty_intersect == False:
                    consecutive_val = float(len(pd.date_range(start=t_start_max, end=t_end_min, freq='15T')))
                    df_to_concat = pd.DataFrame({"BeginDate": t_start_max, "EndDate": t_end_min,
                                                 "Consecutive_Measurements": consecutive_val}, index=[0])
                    t_itv = pd.concat([t_itv, df_to_concat],
                                      ignore_index=True)

    return t_itv.drop_duplicates().reset_index(drop=True)


###################################Tree#######################################
def find_leaves_from_Z(cluster_idx, Z, leaves=[]):
    n_samples = np.shape(Z)[0] + 1
    children_tomerge = Z[cluster_idx][0:2]

    if all(children_tomerge < n_samples):  # tree leaves
        leaves.append(list(children_tomerge))
    elif all(children_tomerge >= n_samples):  # both elements are nodes
        for child in children_tomerge:
            find_leaves_from_Z(int(child - n_samples), Z, leaves)
    else:
        leaves.append(children_tomerge.min())
        find_leaves_from_Z(int(children_tomerge.max()-n_samples), Z, leaves)
    leaves = flatten(leaves)
    leaves = [int(leaf) for leaf in leaves]
    return leaves

def generate_node_keys(n_samples, node_key_ensemble=[]):
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    if len(ascii_uppercase) >= n_samples or len(node_key_ensemble) == 0:
        node_key_ensemble = [key for key in ascii_uppercase[0:n_samples]]
    else:
        node_key_ensemble = [key1 + key2 for key1 in ascii_uppercase for key2 in node_key_ensemble]

    if len(node_key_ensemble) < n_samples:
        node_key_ensemble = generate_node_keys(n_samples, node_key_ensemble)

    # cut out extra values
    node_key_ensemble = node_key_ensemble[0:n_samples]
    return node_key_ensemble

def create_Smatrix_from_linkagematrix(Z: np.array, leaves_labels):
    n_samples = np.shape(Z)[0]+1
    node_key_ensemble = generate_node_keys(n_samples)

    S = np.zeros([n_samples*2-1, n_samples])
    y_ID = [''] * (n_samples-1)

    for i, items in enumerate(Z[::-1]):
        child1, child2, distance, count = items
        children_tomerge = np.array([child1, child2])

        if all(children_tomerge < n_samples):  # tree leaf
            cluster_idx = [int(child) for child in children_tomerge]
        elif all(children_tomerge >= n_samples):  # both elements are nodes
            cluster_idx = find_leaves_from_Z(int(children_tomerge[0]) - n_samples, Z, leaves=[])
            cluster_idx2 = find_leaves_from_Z(int(children_tomerge[1]) - n_samples, Z, leaves=[])
            cluster_idx = cluster_idx + cluster_idx2
        else:  # one element is a node, and one is a leaf
            cluster_idx = find_leaves_from_Z(int(children_tomerge.max()) - n_samples, Z, leaves=[])
            cluster_idx.append(int(children_tomerge.min()))

        # append summation matrix and y vector
        for j in cluster_idx:
            S[i, j] = 1
        y_ID[i] = node_key_ensemble[i]

    # add identity matrix at the bottom of the S matrix
    S[n_samples-1: n_samples*2-1] = np.identity(n_samples)
    y_ID = y_ID + list(leaves_labels)
    return S, y_ID

def cluster_Smatrix_from_threshold(Z: np.array, S: np.array, y_ID: list, threshold: int):

    indexes_to_delete = []
    for i, items in enumerate(Z[::-1]):
        child1, child2, distance, count = items

        if distance < threshold:
            indexes_to_delete.append(i)

    S_clustered = np.delete(S, indexes_to_delete, axis=0)
    y_ID_clustered = [y_id for i, y_id in enumerate(y_ID) if i not in indexes_to_delete]

    return S_clustered, y_ID_clustered


####################################张量乘法########################################
def matrix_tensor_multiplication(matrix, tensor):
    # 获取张量的形状，即第三个维度的大小

    tensor_size = tensor.size(2)

    # 创建一个结果张量来存储结果
    result = torch.zeros(tensor.shape[0], tensor.shape[1], matrix.shape[0])

    # 沿第三个维度将二维矩阵与三维张量相乘
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            # 取出每个1x4的子矩阵，并转置为4x1
            vector = tensor[i, j, :].view(tensor_size, 1)
            # 进行矩阵乘法，结果是4x1，存储在结果张量中
            result[i, j, :] = torch.matmul(matrix, vector).view(-1)

    return result



######################################损失函数###############################3
import torch.nn as nn

class sharq_1(nn.Module):
    def __init__(self):
        super(sharq_1, self).__init__()

    def forward(self,pred, true, S, G, dataset, epsilon=.1):
        SG = torch.matmul(S,G)
        pred_inversed = dataset.inverse_transform_y(pred)
        hier_pred = torch.einsum('ij,bkj->bki', SG, pred_inversed)
        hier_pred = dataset.transform_y(hier_pred)
        reg = epsilon * torch.pow((true - hier_pred), 2)
        reg = torch.mean(reg)
        mse = nn.MSELoss()(pred, true)
        return reg + mse

class sharq_3(nn.Module):
    def __init__(self):
        super(sharq_3, self).__init__()

    def forward(self,pred, pred_Q50, S, G, dataset):
        quantiles_number = pred.size(-1)
        for i in range(quantiles_number):
            pred_q = pred[:,:,:,i]

            SG = torch.matmul(S,G)
            pred_inversed = dataset.inverse_transform_y(pred_q)
            pred_inversed_pred_Q50 = dataset.inverse_transform_y(pred_Q50)
            loss_sharq = torch.pow(pred_inversed-pred_inversed_pred_Q50,2)

            # print(pred_inversed.shape)
            loss_sharq_SG = torch.einsum('ij,bkj->bki', SG, loss_sharq)
            LOSS = torch.pow((dataset.transform_y(dataset.transform_y(loss_sharq_SG-loss_sharq)))/10,2)
            if i==0:
                reg = LOSS
            else:
                reg = reg + LOSS
        reg = torch.mean(reg)
        return reg




####################################结果处理########################################
def visual(true, preds=None, name='./result/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth_Q50', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction_Q50', linewidth=2)
    plt.legend()
    plt.savefig(name, dpi=300, bbox_inches='tight',format="png")
    #plt.show()
    plt.close()


def plot_prediction_interval(data, lower_quantile, upper_quantile, midden_quantile=None, filename='./result/test.pdf'):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='True data', color="black")
    #plt.plot(lower_quantile, label='Lower Quantile', color='orange', linestyle='--')
    #plt.plot(upper_quantile, label='Upper Quantile', color='red', linestyle='--')
    #x = (lower_quantile+upper_quantile)/2
    #plt.plot(midden_quantile, label='midden_quantile', color='yellow', linestyle='--')

    plt.fill_between(range(len(lower_quantile)), lower_quantile.squeeze(), upper_quantile.squeeze(), color='orange',
                     alpha=0.2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Prediction Interval')
    plt.legend()
    plt.savefig(filename)
    #plt.show()
    plt.close()

def adjust_learning_rate(optimizer, epoch,learning_rate, lradj="type1",gamma=0.65):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (gamma ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=6, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def string_split(str_for_split):
    if str_for_split=='': return []
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list

def result_print(result, info_name='Evaluate',folder_path='./result/result_tem/',model_name="1"):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']
    print("========== {} {} results ==========".format(info_name,model_name))
    print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[16], total_MAE[20]))
    print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[4] * 100, total_MAPE[8] * 100, total_MAPE[12] * 100, total_MAPE[16] * 100, total_MAPE[20] * 100))
    print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[4], total_RMSE[8], total_RMSE[12], total_RMSE[16], total_RMSE[20]))
    print("---------------------------------------")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, 'result.txt')
    f = open(file_path, 'a')
    f.write("\n")
    f.write("========== {} {} results ==========".format(info_name,model_name))
    f.write(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f\n" % (
    total_MAE[0], total_MAE[4], total_MAE[8], total_MAE[12], total_MAE[16], total_MAE[20]))
    f.write("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f\n" % (
    total_MAPE[0] * 100, total_MAPE[4] * 100, total_MAPE[8] * 100, total_MAPE[12] * 100, total_MAPE[16] * 100,
    total_MAPE[20] * 100))
    f.write("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f\n" % (
    total_RMSE[0], total_RMSE[4], total_RMSE[8], total_RMSE[12], total_RMSE[16], total_RMSE[20]))
    f.write("---------------------------------------\n")
    
def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(
        times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)


    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        dy = np.abs(diff/y)
        return np.mean(dy)

    for i in range(times):
        y_i = y[:, i, :]
        pred_i = pred[:, i, :]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        MAPE = cal_MAPE(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE
    return result

def quantileLoss(y_true, y_hat, quantiles):
    assert len(y_hat) == len(quantiles)
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_hat[i]
        loss = torch.max((q - 1) * errors, q * errors).mean()
        losses.append(loss)
    return sum(losses)