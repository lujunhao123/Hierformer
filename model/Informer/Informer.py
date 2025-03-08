import torch
import torch.nn as nn

from model.Informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from model.Informer.decoder import Decoder, DecoderLayer
from model.Informer.attn import FullAttention, ProbAttention, AttentionLayer
from model.Informer.embed import DataEmbedding
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
class Model_Q50(nn.Module):
    def __init__(self,arg,distil=True,mix=True,attn='prob'):
        super(Model_Q50, self).__init__()
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
        self.d_model = arg.d_model
        self.y_dim = arg.y_dim
        self.expand = arg.expand
        self.final_layer = nn.Sequential(nn.Linear(arg.node_num, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, self.y_dim)) if self.expand \
            else nn.Linear(arg.node_num, self.y_dim)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        original_batch_size = x_enc.shape[0]
        wind_farms_number = x_enc.shape[2]

        x_enc = rearrange(x_enc, 'b l n d-> (b n) l d')  # BN x L x D
        x_mark_enc = x_mark_enc.repeat(wind_farms_number, 1, 1)
        x_dec = rearrange(x_dec, 'b l n d-> (b n) l d')  # BN x L x D
        x_mark_dec = x_mark_dec.repeat(wind_farms_number, 1, 1)



        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        dec_out = torch.mean(dec_out, dim=-1)[:, :, None]
        predict_y = dec_out[:, -self.pred_len:, :]
        predict_y = rearrange(predict_y, '(b n) l d-> b l (n d)', b=original_batch_size)
        predict_y = self.final_layer(predict_y)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return predict_y, attns
        else:
            return predict_y  # [B, L, D]



class Model_Q05_Q95(nn.Module):
    def __init__(self,arg,distil=True,mix=True,attn='prob'):
        super(Model_Q05_Q95, self).__init__()
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
        self.projection_Q05 = nn.Linear(arg.d_model, arg.c_out, bias=True)
        self.projection_Q95 = nn.Linear(arg.d_model,arg.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        dec_out_Q05 = self.projection_Q05(dec_out)
        dec_out_Q95 = self.projection_Q95(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out_Q05[:, -self.pred_len:, :], dec_out_Q95[:, -self.pred_len:, :], attns


        else:
            return dec_out_Q05[:, -self.pred_len:, :], dec_out_Q95[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    from parma import parse_args
    from hl.utils import string_split
    config = parse_args()
    config.p_hidden_dims = string_split(config.p_hidden_dims)
    config.g_hidden_dims = string_split(config.g_hidden_dims)
    config.drop_columns = string_split(config.drop_columns)
    config.data_split = string_split(config.data_split)
    x = torch.randn((256, 96, 6, 5))
    y = torch.randn((256, 48, 6, 5))
    xt = torch.randn((256, 96, 5))
    yt = torch.randn((256, 48, 5))
    graph = torch.randn((256, 6, 6))
    model = Model_Q50(config)
    output_1 = model(x,xt,y,yt)
    print(output_1.shape)
    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))