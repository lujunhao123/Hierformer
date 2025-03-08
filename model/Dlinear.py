import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model_Q50(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, config):
        super(Model_Q50, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = True
        self.channels = config.data_dim
        self.y_dim = config.y_dim
        self.expand = config.expand
        self.d_model = config.d_model

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.final_layer = nn.Sequential(nn.Linear(config.node_num, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, self.y_dim)) if self.expand \
            else nn.Linear(config.node_num, self.y_dim)


    def forward(self, x, graph=None):
        # x: [Batch, Input length, Channel]
        original_batch_size = x.shape[0]
        x = rearrange(x, 'b l n d-> (b n) l d')  # BN x L x D

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            # print(seasonal_output.shape)

            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
                # print(seasonal_output.shape)

        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1).contiguous()
        x = torch.mean(x,dim=-1)[:,:,None]
        # print("x",x.shape)
        predict_y = rearrange(x, '(b n) l d-> b l (n d)', b=original_batch_size)
        predict_y = self.final_layer(predict_y)
        return predict_y  # to [Batch, Output length, Channel]



class Dlinear_Q05_Q95(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len,pred_len,input_dim):
        super(Dlinear_Q05_Q95, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = True
        self.channels = input_dim

        if self.individual:
            self.Linear_Seasonal_Q05 = nn.ModuleList()
            self.Linear_Seasonal_Q95 = nn.ModuleList()
            self.Linear_Trend_Q05 = nn.ModuleList()
            self.Linear_Trend_Q95 = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal_Q05.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal_Q95.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend_Q05.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend_Q95.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output_Q05 = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)

            seasonal_output_Q95 = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)


            trend_output_Q05 = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)

            trend_output_Q95 = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            # print(seasonal_output.shape)

            for i in range(self.channels):
                seasonal_output_Q05[:, i, :] = self.Linear_Seasonal_Q05[i](seasonal_init[:, i, :])
                seasonal_output_Q05[:, i, :] = self.Linear_Seasonal_Q95[i](seasonal_init[:, i, :])
                trend_output_Q05[:, i, :] = self.Linear_Trend_Q05[i](trend_init[:, i, :])
                trend_output_Q95[:, i, :] = self.Linear_Trend_Q95[i](trend_init[:, i, :])
                # print(seasonal_output.shape)


        x_Q05 = seasonal_output_Q05 + trend_output_Q05
        x_Q95 = seasonal_output_Q95 + trend_output_Q95
        x.contiguous().view(-1)
        return x_Q05.permute(0, 2, 1).contiguous(), x_Q95.permute(0, 2, 1).contiguous() # to [Batch, Output length, Channel]





if __name__ == "__main__":
    from parma import parse_args

    config = parse_args()
    x_seq = torch.randn((256, 96, 6, 5))
    print(x_seq.shape)
    graph = torch.randn((256, 6, 6))
    model = Model_Q50(config)
    output = model(x_seq,graph)
    print(output.shape)

