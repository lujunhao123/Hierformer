import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from model.must.Crossformer.cross_encoder import Encoder
from model.must.Crossformer.cross_decoder import Decoder
from model.must.Crossformer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from model.must.Crossformer.cross_embed import DSW_embedding

from math import ceil


class _Model_Q50(nn.Module):
    def __init__(self,config):
        super(_Model_Q50, self).__init__()
        self.data_dim = config.data_dim
        self.in_len = config.in_len
        self.out_len = config.out_len
        self.seg_len = config.seg_len
        self.merge_win = config.win_size
        self.d_model = config.d_model
        self.y_dim = config.hier_number


        self.device = config.device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * config.in_len / config.seg_len) * config.seg_len
        self.pad_out_len = ceil(1.0 * config.out_len / config.seg_len) * config.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(config.seg_len, config.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, config.data_dim, (self.pad_in_len // config.seg_len), config.d_model))
        self.pre_norm = nn.LayerNorm(config.d_model)

        # Encoder
        self.encoder = Encoder(config.e_layers, config.win_size, config.d_model, config.n_heads, config.d_ff, block_depth=1, \
                               dropout=config.dropout, in_seg_num=(self.pad_in_len // config.seg_len), factor=config.factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, config.data_dim, (self.pad_out_len // config.seg_len), config.d_model))
        self.decoder = Decoder(config.seg_len, config.e_layers + 1, config.d_model, config.n_heads, config.d_ff, config.dropout, \
                               out_seg_num=(self.pad_out_len // config.seg_len), factor=config.factor)
        self.d_model = config.d_model
        self.final_layer = nn.Sequential(nn.Linear(config.data_dim, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, self.y_dim))

        # Multiquantile prediction
        self.loss_function = config.loss_fun
        self.device = config.device
        self.output_dim = config.output_dim
        self.d_model = config.d_model
        self.mlp = nn.Sequential(nn.Linear(1, self.d_model // 2), nn.GELU(),
                                 nn.Linear(self.d_model // 2, self.output_dim))

    def forward(self, x_seq, x_mark_enc, x_dec, x_mark_dec,tau,tau_norm):

        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq)

        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        dec_out = self.decoder(dec_in, enc_out)
        predict_y = dec_out[:, :self.out_len, :]
        predict_y = self.final_layer(predict_y)

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
