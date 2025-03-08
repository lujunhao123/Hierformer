import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, out_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """

        return self.network(x)


class _Model_Q50(nn.Module):
    def __init__(self,configs):
        super(_Model_Q50, self).__init__()
        self.tcn = TCN(num_inputs=configs.enc_in, num_channels=[32, 32, 32, 32])
        self.fc = nn.Linear(32,configs.hier_number)
        self.pred_len = configs.pred_len

        # Multiquantile prediction
        self.loss_function = configs.loss_fun
        self.device = configs.device
        self.output_dim = configs.output_dim
        self.mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(),
                                 nn.Linear(32, self.output_dim))


    def forward(self, x, x_mark_enc, x_dec, x_mark_dec,tau,tau_norm):
        x = x.permute(0,2,1)      # batch_size,seq_length,in_feature -- batch_size,in_feature,seq_length
        x = self.tcn(x)
        x = x.permute(0, 2, 1)    # batch_size,hidden_dim,seq_length--batch_size,seq_length,hidden_dim
        x = self.fc(x)          # batch_size,seq_length,hidden_dim--batch_size,seq_length,out_dim

        predict_y = x[:, -self.pred_len:, :]

        if self.loss_function == 'mse':
            return predict_y


        elif self.loss_function == "htqf" or self.loss_function == "htqf_hier":
            # Multiquantile prediction
            q_input = predict_y[:, :, :, None]
            output = self.mlp(q_input)

            mu = output[..., 0:1]
            sig = output[..., 1:2]
            utail = output[..., 2:3]
            vtail = output[..., 3:4]

            tau_tensor = torch.tensor(tau, device=self.device)
            tau_norm_tensor = torch.tensor(tau_norm, device=self.device)

            factor1 = torch.exp(torch.multiply(utail, tau_norm_tensor)) / 4 + 1
            factor2 = torch.exp(-torch.multiply(vtail, tau_norm_tensor)) / 4 + 1
            factor = torch.multiply(factor1, factor2)
            q = torch.add(torch.multiply(torch.multiply(factor, tau_norm_tensor), sig), mu)
            return q

if __name__ == '__main__':
    from parma_1 import parse_args
    import numpy as np
    import scipy

    tau = np.r_[np.linspace(0.025, 0.075, 3), np.linspace(0.1, 0.9, 9), np.linspace(0.925, 0.975, 3)].reshape(1, -1)
    tau_norm = scipy.stats.norm.ppf(tau)
    config = parse_args()
    model = _Model_Q50(config)
    x = torch.randn((32,96,9))
    y = torch.randn((32, 32 + 24, 9))
    xt = torch.randn((32, 96, 5))
    yt = torch.randn((32, 32 + 24, 5))
    graph = torch.randn((15,9))
    ouput = model(x,xt,y,yt,tau,tau_norm)
    print(ouput.shape)