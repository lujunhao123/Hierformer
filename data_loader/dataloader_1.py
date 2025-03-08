import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from hl.timefeatures import time_features

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self,data):
        if type(data) is list:
            data = [item for sublist in data for item in sublist]
            data = np.array(data)
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



class Wind_Dataset(Dataset):
    def __init__(self,root_path, data_S_path='NSW1_S.npy',data_G_path="NSW1_G.npy",flag='train',
                 hier_df=None,size=None,scale=True,data_split=[0.7, 0.1, 0.2],timeenc=None,freq=None,site=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_S_path = data_S_path
        self.data_G_path = data_G_path
        self.data_split = data_split
        self.scale = scale
        self.hier_df = hier_df
        self.timeenc = timeenc
        self.freq = freq

        self.__read_data__(site)


    def __read_data__(self,site):
        df = self.hier_df
        if site == "NSW1":
            df_leaves = df.iloc[:,-9:]
            self.matrix_S = np.load(os.path.join(self.root_path,
                                                 self.data_S_path))
            self.matrix_G = np.load(os.path.join(self.root_path,
                                                 self.data_G_path))
        elif site == "SA1":
            df_leaves = df.iloc[:,-20:]
            self.matrix_S = np.load(os.path.join(self.root_path,
                                                 self.data_S_path))
            self.matrix_G = np.load(os.path.join(self.root_path,
                                                 self.data_G_path))
        # 划分数据集
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]
            val_num = self.data_split[1]
            test_num = self.data_split[2]

        train_num = int(len(df_leaves) * self.data_split[0])
        test_num = int(len(df_leaves) * self.data_split[2])
        val_num = len(df_leaves) - train_num - test_num
        border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
        border2s = [train_num, train_num + val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data_x = df_leaves[border1s[0]:border2s[0]].values
        data_x = df_leaves[border1:border2].values
        data_xforq = df.iloc[:,1:][border1:border2].values
        self.scaler_x = StandardScaler(train_data_x)
        if self.scale == True:
            scaler_data_x = self.scaler_x.transform(data_x)
            data_x = scaler_data_x
        self.data_x = data_x

        train_data_y = df.iloc[border1s[0]:border2s[0], 1:].values
        data_y = df.iloc[border1:border2,1:].values
        self.scaler_y = StandardScaler(train_data_y)
        if self.scale == True:
            scaler_data_y = self.scaler_y.transform(data_y)
            data_y = scaler_data_y

            scaler_data_xforq = self.scaler_y.transform(data_xforq)
            data_xforq = scaler_data_xforq
        self.data_y = data_y
        self.data_xforq = data_xforq

        # 处理时间
        df_stamp = df[['date']][border1:border2]

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        l_begin = s_end - self.label_len
        l_end = l_begin + self.label_len + self.pred_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_y = self.data_x[l_begin:l_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_y_mark = self.data_stamp[l_begin:l_end]
        seq_y = self.data_y[r_begin:r_end]
        matrix_S = self.matrix_S
        matrix_G = self.matrix_G
        seq_xforq = self.data_xforq[s_begin:s_end]

        return seq_x,seq_x_y,seq_x_mark,seq_x_y_mark,seq_y,matrix_S,matrix_G,seq_xforq

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_data_x_sta(self):
        return self.scaler_x.mean, self.scaler_x.std

    def get_data_y_sta(self):
        return self.scaler_y.mean, self.scaler_y.std

    def transform_x(self, data):
        data = self.scaler_x.transform(data)
        return data

    def inverse_transform_x(self,data):
        data_inversed = self.scaler_x.inverse_transform(data)
        return data_inversed

    def inverse_transform_y(self,data):
        data_inversed = self.scaler_y.inverse_transform(data)
        return data_inversed

    def transform_y(self, data):
        data = self.scaler_y.transform(data)
        return data