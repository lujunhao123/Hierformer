import torch
import torch.nn as nn
from model.must.Autoformer.Embed import DataEmbedding_wo_pos
from model.must.Autoformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from model.must.Autoformer.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from einops import rearrange, repeat



"""

def __init__(self,enc_in,dec_in,c_out,seq_len,label_len,pred_len,d_model=128,d_ff=256,dropout=0.1,n_heads=8,
                 moving_avg=25,e_layer=2,d_layer=1,factor=3):
"""

"""
def __init__(self,enc_in,dec_in,c_out,seq_len,label_len,pred_len,d_model,d_ff,dropout,n_heads,
                 moving_avg,e_layer,d_layer,factor
"""
class _Model_Q50(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self,arg):
        super(_Model_Q50, self).__init__()
        self.seq_len = arg.seq_len
        self.label_len = arg.label_len
        self.pred_len = arg.pred_len
        self.output_attention = False

        # Decomp
        kernel_size = arg.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(arg.enc_in, arg.d_model,arg.embed, arg.freq,
                                                  arg.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(arg.dec_in, arg.d_model, arg.embed,  arg.freq,
                                                  arg.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, arg.factor, attention_dropout=0.1,
                                        output_attention=False),
                        arg.d_model, arg.n_heads),
                    arg.d_model,
                    arg.d_ff,
                    moving_avg=arg.moving_avg,
                    dropout=arg.dropout,
                    activation='gelu'
                ) for l in range(arg.e_layers)
            ],
            norm_layer=my_Layernorm(arg.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, arg.factor, 0.1,
                                        output_attention=False),
                        arg.d_model, arg.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False,arg.factor, attention_dropout=0.1,
                                        output_attention=False),
                        arg.d_model, arg.n_heads),
                    arg.d_model,
                    arg.c_out,
                    arg.d_ff,
                    moving_avg=arg.moving_avg,
                    dropout=arg.dropout,
                    activation='gelu',
                )
                for l in range(arg.d_layers)
            ],
            norm_layer=my_Layernorm(arg.d_model),
            projection=nn.Linear(arg.d_model, arg.c_out, bias=True)
        )

        self.d_model = arg.d_model
        self.y_dim = arg.hier_number
        self.final_layer = nn.Sequential(nn.Linear(arg.data_dim, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, self.y_dim))

        # Multiquantile prediction
        self.loss_function = arg.loss_fun
        self.device = arg.device
        self.output_dim = arg.output_dim
        self.d_model = arg.d_model
        self.mlp = nn.Sequential(nn.Linear(1, self.d_model//2), nn.GELU(),
                                 nn.Linear(self.d_model//2, self.output_dim))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,tau,tau_norm,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)

        # final
        dec_out = trend_part + seasonal_part
        predict_y = dec_out[:, -self.pred_len:, :]
        predict_y = self.final_layer(predict_y)

        if self.loss_function == 'mse':
            return predict_y

        elif self.loss_function == "htqf" or self.loss_function == "htqf_hier":
            # Multiquantile prediction
            q_input = predict_y[:,:,:,None]
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