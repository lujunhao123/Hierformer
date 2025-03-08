import torch
import torch.nn as nn

from model.ST_htqf_hier.Informer.Informer import _Model_Q50

from model.ST_htqf_hier.Graph_Encoder.graph_encoder import GraphEncoder
import numpy as np

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.Informer = _Model_Q50(config)
        self.pred_len = config.pred_len
        self.loss_fun = config.loss_fun
        self.device = config.device

    def forward(self,seq_x, seq_x_mark, dec_y, seq_x_y_mark,tau,tau_norm, matrix_S,dataset,matrix_G):
        pred_out = self.Informer(seq_x, seq_x_mark, dec_y, seq_x_y_mark,tau,tau_norm, matrix_S,dataset)

        if self.loss_fun == "mse":
            pred_out = dataset.inverse_transform_y(pred_out)
            hier_out = torch.matmul(pred_out, matrix_S[0].T)
            hier_out = dataset.transform_y(hier_out)
            return hier_out

        elif self.loss_fun == "htqf" or self.loss_fun == "htqf_hier":
            S = matrix_S[0].detach().cpu().numpy()
            G = matrix_G[0].detach().cpu().numpy()
            SG = np.matmul(S, G)
            SG_T = torch.tensor(SG)
            SG_T = SG_T.float().to(self.device)
            output_inversed = dataset.inverse_transform_y(pred_out)
            output_inversed = output_inversed.float()
            output_must = torch.einsum('ij,bkjf->bkif', SG_T, output_inversed)
            pred_out = dataset.transform_y(output_must)
            return pred_out



if __name__ == '__main__':
    from parma_1 import parse_args
    config = parse_args()
    model = Model(config)
    x = torch.randn((32,96,9))
    y = torch.randn((32, 32 + 24, 9))
    xt = torch.randn((32, 96, 5))
    yt = torch.randn((32, 32 + 24, 5))
    graph = torch.randn((15,9))
    ouput = model(x,xt,y,yt,graph)
    print(ouput.shape)