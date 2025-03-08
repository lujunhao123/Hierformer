import torch
import torch.nn as nn


class _Model_Q50(nn.Module):
    def __init__(self,configs):
        super(_Model_Q50, self).__init__()
        self.lstm = nn.LSTM(input_size=configs.enc_in, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64,configs.hier_number)
        self.pred_len = configs.pred_len

        # Multiquantile prediction
        self.loss_function = configs.loss_fun
        self.device = configs.device
        self.output_dim = configs.output_dim
        self.mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(),
                                 nn.Linear(32, self.output_dim))


    def forward(self,x, x_mark_enc, x_dec, x_mark_dec,tau,tau_norm):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        x,_ = self.lstm(x,(h0,c0))
        x = self.fc(x)

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