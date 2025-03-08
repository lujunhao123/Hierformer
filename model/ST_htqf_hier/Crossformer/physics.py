import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

###HSPGCN
class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt):
        super(T_cheby_conv_ds, self).__init__()
        c_in_new = (K) * 1
        self.conv1 = Conv2d(1, 1, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.conv2 = Conv2d(2, 1, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.conv3 = Conv2d(c_in_new, 1, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.conv4 = Conv2d(1, 1, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        # calculat one-order laplacian matrix
        Ls = []
        L1 = adj
        # L0 = torch.eye(nNode).repeat(nSample, 1, 1).cuda()
        L0 = torch.eye(nNode).repeat(nSample, 1, 1).to("cuda" if torch.cuda.is_available() else "cpu")
        Ls.append(L0)
        Ls.append(L1)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        Lap = Lap.transpose(-1, -2)

        # calculat one-order laplacian convolution
        x2 = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x2 = x2.view(nSample, -1, nNode, length)
        out2 = self.conv2(x2)

        time_toeplitz_matrix = np.eye(nNode)
        for ii in range(nNode - 1):
            time_toeplitz_matrix[ii, ii + 1] = -1
        total_time_toeplitz_matrix = np.zeros(([Lap.shape[0], 1, Lap.shape[2], Lap.shape[3]]))

        II = np.eye(nNode)
        III = np.zeros(([Lap.shape[0], 1, Lap.shape[2], Lap.shape[3]]))

        for jj in range(Lap.shape[0]):
            total_time_toeplitz_matrix[jj, :, :, :] = time_toeplitz_matrix
            III[jj, :, :, :] = II

        Toeplitz = torch.tensor(total_time_toeplitz_matrix, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
        IIII = torch.tensor(III, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

        # calculat zero-order laplacian convolution
        x1 = torch.einsum('bcnl,bknq->bckql', x, IIII).contiguous()
        x1 = x1.view(nSample, -1, nNode, length)
        out1 = self.conv1(x1)

        # calculat Toeplitz convolution
        x4 = torch.einsum('bcnl,bknq->bckql', x, Toeplitz).contiguous()
        x4 = x4.view(nSample, -1, nNode, length)
        out4 = self.conv4(x4)

        out = out2 + out1 + out4
        return out

class SATT_0(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(SATT_0, self).__init__()
        self.conv1 = Conv2d(1, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(num_nodes)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 2, 3)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n
        c2 = seq.permute(0, 3, 1, 2)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(1)  # b,c,n
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs


class PHYSICS_LAYER(nn.Module):
    def __init__(self,c_out, num_nodes, tem_size, K, Kt):
        super(PHYSICS_LAYER, self).__init__()
        self.SATT_0 = SATT_0(c_out, num_nodes, tem_size)
        self.dynamic_gcn = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt)
        self.K = K
        self.tem_size = tem_size
        self.c_out = c_out

    def forward(self, x, supports):
        S_coef = self.SATT_0(x)
        adj_out = S_coef
        adj_out1 = adj_out * supports
        x_1 = self.dynamic_gcn(x, adj_out1)
        out = F.leaky_relu(x_1)

        return out,adj_out1


class HSPGNN(nn.Module):
    def __init__(self, c_out, num_nodes, tem_size, K, Kt):
        super(HSPGNN, self).__init__()
        self.layer1 = PHYSICS_LAYER(c_out, num_nodes, tem_size, K, Kt)
        self.layer2 = PHYSICS_LAYER(c_out, num_nodes, tem_size, K, Kt)

    def forward(self, x, supports):
        x,supports = self.layer1(x,supports)
        output, supports = self.layer2(x,supports)
        return output


if __name__ == '__main__':
    model = HSPGNN(1,7,96,3,3)
    input = torch.randn((256,1,7,96))
    support = torch.randn((7,7))
    out = model(input, support)
    print(out.shape)