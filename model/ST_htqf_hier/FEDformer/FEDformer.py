import torch
import torch.nn as nn
import torch.nn.functional as F
from model.must.FEDformer.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp
from model.must.FEDformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from model.must.FEDformer.FourierCorrelation import FourierBlock, FourierCrossAttention
from model.must.FEDformer.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from model.must.FEDformer.SelfAttention_Family import FullAttention, ProbAttention
# from layers.FED_wo_decomp import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from model.must.FEDformer.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np
from einops import rearrange, repeat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""

def __init__(self, enc_in, dec_in, c_out,seq_len,label_len,pred_len,d_model=128,d_ff=256,
                 dropout=0.05, n_heads=8,moving_avg=25,e_layers=2,d_layers=1,
                 version='Fourier',mode_select ='random',modes =64,
                 output_attention=False,freq ='h',embed_type=0,
                 activation='gelu',embed='timeF'):
                 
                 """

"""
def __init__(self, enc_in, dec_in,c_out,seq_len,label_len,pred_len,d_model,d_ff,
                 dropout,n_heads,moving_avg,e_layers,d_layers,
                 version,mode_select,modes,
                 output_attention,freq,embed_type,
                 activation,embed):
"""

class _Model_Q50(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self,arg):
        super(_Model_Q50, self).__init__()

        self.seq_len = arg.seq_len
        self.label_len = arg.label_len
        self.pred_len = arg.pred_len

        self.enc_in = arg.enc_in
        self.dec_in = arg.dec_in
        self.version = arg.version
        self.output_attention = arg.output_attention

        self.y_dim = arg.hier_number
        self.d_model = arg.d_model
        self.final_layer = nn.Sequential(nn.Linear(arg.enc, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, self.y_dim))

        # Decomp
        kernel_size = arg.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        # self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                           configs.dropout)
        # self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                           configs.dropout)
        if arg.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)
        elif arg.embed_type == 1:
            self.enc_embedding = DataEmbedding(self.enc_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)
            self.dec_embedding = DataEmbedding(self.dec_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)
        elif arg.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos_temp(self.enc_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(self.dec_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)
        elif arg.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(self.enc_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(self.dec_in, arg.d_model, arg.embed, arg.freq,
                                                    arg.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=arg.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=arg.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=arg.d_model,
                                                  out_channels=arg.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=arg.modes,
                                                  ich=arg.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=arg.d_model,
                                            out_channels=arg.d_model,
                                            seq_len=self.seq_len,
                                            modes=arg.modes,
                                            mode_select_method=arg.mode_select)
            decoder_self_att = FourierBlock(in_channels=arg.d_model,
                                            out_channels=arg.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=arg.modes,
                                            mode_select_method=arg.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=arg.d_model,
                                                      out_channels=arg.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=arg.modes,
                                                      mode_select_method=arg.mode_select)
        # Encoder
        enc_modes = int(min(arg.modes, self.seq_len//2))
        dec_modes = int(min(arg.modes, (self.seq_len//2+self.pred_len)//2))
        #print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        arg.d_model, arg.n_heads),

                    arg.d_model,
                    arg.d_ff,
                    moving_avg=arg.moving_avg,
                    dropout=arg.dropout,
                    activation=arg.activation
                ) for l in range(arg.e_layers)
            ],
            norm_layer=my_Layernorm(arg.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        arg.d_model, arg.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        arg.d_model, arg.n_heads),
                    arg.d_model,
                    arg.c_out,
                    arg.d_ff,
                    moving_avg=arg.moving_avg,
                    dropout=arg.dropout,
                    activation=arg.activation,
                )
                for l in range(arg.d_layers)
            ],
            norm_layer=my_Layernorm(arg.d_model),
            projection=nn.Linear(arg.d_model, arg.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):


        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
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
        return predict_y

class Model_multiQ(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, arg):
        super(Model_multiQ, self).__init__()
        self.Model_Q50 = _Model_Q50(arg)
        self.multi_Q = nn.ModuleList([nn.Linear(arg.hier_number, arg.hier_number) for _ in range(8)])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        output = self.Model_Q50(x_enc, x_mark_enc, x_dec, x_mark_dec)
        outputs = []

        for i in range(8):
            output_q = self.multi_Q[i](output)
            outputs.append(output_q.unsqueeze(-1))
        combined_output = torch.cat(outputs, dim=-1)

        # [B,L,D,8]
        return output,combined_output



if __name__ == '__main__':
    from parma import parse_args
    from hl.utils import string_split
    config = parse_args()
    config.p_hidden_dims = string_split(config.p_hidden_dims)
    config.g_hidden_dims = string_split(config.g_hidden_dims)
    config.drop_columns = string_split(config.drop_columns)
    config.data_split = string_split(config.data_split)
    x = torch.randn((256, 96, 6, 5))
    y = torch.randn((256, 32+24, 6, 5))
    xt = torch.randn((256, 96, 5))
    yt = torch.randn((256, 32+24, 5))
    graph = torch.randn((256, 6, 6))
    model = _Model_Q50(config)
    output_1 = model(x,xt,y,yt)
    print(output_1.shape)
    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))