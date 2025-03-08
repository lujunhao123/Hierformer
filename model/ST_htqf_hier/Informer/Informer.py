import torch
import torch.nn as nn

from model.ST_htqf_hier.Informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from model.ST_htqf_hier.Informer.decoder import Decoder, DecoderLayer
from model.ST_htqf_hier.Informer.attn import FullAttention, ProbAttention, AttentionLayer
from model.ST_htqf_hier.Informer.embed import DataEmbedding
from einops import rearrange, repeat


"""

def __init__(self, enc_in, c_out,pred_len,
                 factor, d_model, n_heads, e_layers, d_layers, d_ff,
                 dropout, attn='prob', embed='timeF', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 ):
                 
"""

"""
self, enc_in,dec_in,c_out,pred_len,
                 factor, d_model, n_heads, e_layers, d_layers, d_ff,
                 dropout,embed, freq, activation,
                 output_attention,distil=True,mix=True,attn='prob'
"""
class _Model_Q50(nn.Module):
    def __init__(self,arg,distil=True,mix=True,attn='prob'):
        super(_Model_Q50, self).__init__()
        self.pred_len = arg.pred_len
        self.attn = attn
        self.output_attention = arg.output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(arg.enc_in, arg.d_model, arg.embed, arg.freq, arg.dropout)
        self.dec_embedding = DataEmbedding(arg.dec_in, arg.d_model, arg.embed, arg.freq, arg.dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, arg.factor, attention_dropout=arg.dropout, output_attention=arg.output_attention),
                                   arg.d_model, arg.n_heads, mix=False),
                    arg.d_model,
                    arg.d_ff,
                    dropout=arg.dropout,
                    activation=arg.activation
                ) for l in range(arg.e_layers)
            ],
            [
                ConvLayer(
                    arg.d_model
                ) for l in range(arg.e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(arg.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, arg.factor, attention_dropout=arg.dropout, output_attention=False),
                                   arg.d_model, arg.n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, arg.factor, attention_dropout=arg.dropout, output_attention=False),
                                   arg.d_model, arg.n_heads, mix=False),
                    arg.d_model,
                    arg.d_ff,
                    dropout=arg.dropout,
                    activation=arg.activation,
                )
                for l in range(arg.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(arg.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(arg.d_model, arg.c_out, bias=True)

        # Multiquantile prediction
        self.loss_function = arg.loss_fun
        self.device = arg.device
        self.output_dim = arg.output_dim
        self.d_model = arg.d_model
        self.final_layer = nn.Sequential(nn.Linear(arg.data_dim, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, arg.hier_number))

        self.mlp = nn.Sequential(nn.Linear(1, self.d_model // 2), nn.GELU(),
                                 nn.Linear(self.d_model // 2, 4))


        self.data_dim = arg.data_dim

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,tau,tau_norm, matrix_S,dataset,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):


        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        predict_y = dec_out[:, -self.pred_len:, :]

        if self.loss_function == 'mse':
            return predict_y


        elif self.loss_function == "htqf" or self.loss_function=="htqf_hier":
            predict_y = self.final_layer(predict_y)
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