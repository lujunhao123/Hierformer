import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from model.ST_htqf_hier.Crossformer.cross_encoder import Encoder
from model.ST_htqf_hier.Crossformer.cross_decoder import Decoder
from model.ST_htqf_hier.Crossformer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from model.ST_htqf_hier.Crossformer.cross_embed import DSW_embedding

from math import ceil


class Model_Q50(nn.Module):
    def __init__(self,config):
        super(Model_Q50, self).__init__()
        self.data_dim = config.data_dim
        self.in_len = config.in_len
        self.out_len = config.out_len
        self.seg_len = config.seg_len
        self.merge_win = config.win_size
        self.d_model = config.d_model
        self.y_dim = config.y_dim
        self.expand = config.expand


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

        # hier
        self.final_layer = nn.Sequential(nn.Linear(config.node_num, self.d_model), nn.GELU(),
                                         nn.Linear(self.d_model, self.y_dim)) if self.expand \
            else nn.Linear(config.node_num, self.y_dim)

        # Channel-wise LoRA
        self.rank = 16
        self.node__dim = 32
        # Embedding
        self.embed_dim = config.d_model - self.node__dim
        self.enc_value_embedding = DSW_embedding(config.seg_len, self.embed_dim)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, config.data_dim, (self.pad_in_len // config.seg_len), self.embed_dim))
        self.pre_norm = nn.LayerNorm(self.embed_dim)
        self.adapter = nn.Parameter(torch.empty(config.data_dim, 16, self.embed_dim, self.rank))  # [n,seg_num,D,d]
        nn.init.xavier_uniform_(self.adapter)
        self.lora = nn.Linear(self.rank, self.node__dim, bias=False)


    def forward(self, x_seq, graph=None):

        original_batch_size = x_seq.shape[0]
        x_seq = rearrange(x_seq, 'b l n d-> (b n) l d')  # BN x L x D
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq)
        # print("123",x_seq.shape)       # [32,7,28,256,,batch_size,data_dim,seg_number,d_model]
        # print("112",1,self.data_dim,(self.pad_in_len // self.seg_len),self.d_model)  #[1,7,28,256]
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        # Local adaptation
        B, _, _, N = x_seq.shape
        adaptation = []
        adapter = F.relu(self.lora(self.adapter))  # [n s D d]
        adapter = adapter.permute(1, 2, 3, 0)  # [n s D d] -> [s D d n]
        adapter = repeat(adapter, 's D d n -> repeat s D d n', repeat=B)  # [b, s, D, d, n]
        x_seq = x_seq.permute(0, 2, 3, 1)  # [b n s D] -> [b s D n]
        adaptation.append(torch.einsum('bsDn,bsDdn->bsdn', [x_seq, adapter]))  # [B, d, N]
        x_seq = torch.cat([x_seq] + adaptation, dim=2)  # [B, H', N]
        x_seq = x_seq.permute(0, 3, 1, 2)  # [b s D n] -> [b n s D]
        # Local adaptation

        enc_out = self.encoder(x_seq)
        # print(x_seq.shape, "//", base)
        # print(type(enc_out)) # list
        # print(len(enc_out))  # 4
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        dec_out = self.decoder(dec_in, enc_out)
        dec_out = torch.mean(dec_out,dim=-1)[:,:,None]
        predict_y = dec_out[:, :self.out_len, :]
        predict_y = rearrange(predict_y, '(b n) l d-> b l (n d)', b=original_batch_size)
        predict_y = self.final_layer(predict_y)
        return predict_y

if __name__ == '__main__':
    from parma import parse_args
    from hl.utils import string_split
    config = parse_args()
    config.p_hidden_dims = string_split(config.p_hidden_dims)
    config.g_hidden_dims = string_split(config.g_hidden_dims)
    config.drop_columns = string_split(config.drop_columns)
    config.data_split = string_split(config.data_split)
    x_seq = torch.randn((256, 96, 6, 5))
    print(x_seq.shape)
    graph = torch.randn((256, 6, 6))
    model = Model_Q50(config)
    output_1 = model(x_seq)
    print(output_1.shape)
    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
