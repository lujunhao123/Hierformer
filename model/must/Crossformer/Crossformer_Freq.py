import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from model.ST_htqf_hier.Crossformer.cross_encoder import Encoder
from model.ST_htqf_hier.Crossformer.cross_decoder_freq import Decoder
from model.ST_htqf_hier.Crossformer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from model.ST_htqf_hier.Crossformer.cross_embed import DSW_embedding

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
        self.enc_value_embedding_z1 = DSW_embedding(config.seg_len, config.d_model)
        self.enc_value_embedding_z2 = DSW_embedding(config.seg_len, config.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, config.data_dim, (self.pad_in_len // config.seg_len), config.d_model*2))
        self.pre_norm = nn.LayerNorm(config.d_model*2)

        # Encoder
        self.encoder = Encoder(config.e_layers, config.win_size, config.d_model*2, config.n_heads, config.d_ff*2, block_depth=1, \
                               dropout=config.dropout, in_seg_num=(self.pad_in_len // config.seg_len), factor=config.factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, config.data_dim, (self.pad_out_len // config.seg_len), config.d_model*2))
        self.decoder = Decoder(config.seg_len, config.e_layers + 1, config.d_model*2, config.n_heads, config.d_ff, config.dropout, \
                               out_seg_num=(self.pad_out_len // config.seg_len), factor=config.factor)


        # Multiquantile prediction
        self.loss_function = config.loss_fun
        self.device = config.device
        self.output_dim = config.output_dim
        self.d_model = config.d_model
        self.final_layer = nn.Sequential(nn.Linear(config.data_dim, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, config.hier_number))

        self.mlp = nn.Sequential(nn.Linear(1, self.d_model // 2), nn.GELU(),
                                 nn.Linear(self.d_model // 2,4))


        self.data_dim = config.data_dim

        self.getr1 = nn.Sequential(nn.Linear(config.d_model*2,config.d_model*2),
                                   nn.Linear(config.d_model*2,config.seg_len))
        self.getr2 = nn.Sequential(nn.Linear(config.d_model*2,config.d_model*2),
                                nn.Linear(config.d_model*2,config.seg_len))
        self.ircom = nn.Linear(config.pred_len * 2, config.pred_len)

    def forward(self, x_seq, seq_x_mark, dec_y, seq_x_y_mark,tau,tau_norm):

        batch_size,seq_len,enc_in = x_seq.shape
        z = torch.fft.fft(x_seq)
        z1 = z.real
        z2 = z.imag
        if (self.in_len_add != 0):
            z1 = torch.cat((z1[:, :1, :].expand(-1, self.in_len_add, -1), z1), dim=1)
            z2 = torch.cat((z2[:, :1, :].expand(-1, self.in_len_add, -1), z2), dim=1)

        z1 = self.enc_value_embedding_z1(z1)
        z2 = self.enc_value_embedding_z2(z2)

        seq_x = torch.cat((z1,z2),-1)
        seq_x += self.enc_pos_embedding
        seq_x = self.pre_norm(seq_x )

        enc_out = self.encoder(seq_x)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        z = self.decoder(dec_in, enc_out)  # b (out_d seg_num) d_model
        z1 = self.getr1(z)
        z2 = self.getr2(z)
        z1 = rearrange(z1, 'b (out_d seg_num) seg_len -> b out_d (seg_num seg_len) ', out_d = enc_in)
        z2 = rearrange(z2, 'b (out_d seg_num) seg_len -> b out_d (seg_num seg_len) ', out_d= enc_in)
        z = torch.fft.ifft(torch.complex(z1, z2))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))
        predict_y = z.permute(0, 2, 1)

        predict_y = self.final_layer(predict_y)
        if self.loss_function == 'mse':
            return predict_y

        elif self.loss_function == "htqf" or self.loss_function=="htqf_hier":
            q_input = predict_y[:, :, :, None]
            output = self.mlp(q_input)

            mu = output[..., 0:1]
            sig = output[..., 1:2]
            utail = output[..., 2:3]
            vtail = output[..., 3:4]

            tau_tensor = torch.tensor(tau, device=self.device).float()
            tau_norm_tensor = torch.tensor(tau_norm, device=self.device).float()

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
