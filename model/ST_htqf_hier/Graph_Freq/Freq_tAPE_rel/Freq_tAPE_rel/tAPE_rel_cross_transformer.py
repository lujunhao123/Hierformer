import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
from embed import tAPE
from torch import einsum
from einops import rearrange
class Cross_Transformer(nn.Module):
    def __init__(self,
                 patch_dim,patch_num,
                 cf_dim,cf_mlp,cf_depth,heads,
                 cf_dim_head,d_model,dropout,use_pos,use_att_rel
                 ):

        self.use_pos = use_pos
        self.use_att_rel = use_att_rel
        super(Cross_Transformer, self).__init__()
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, cf_dim),nn.Dropout(dropout))
        self.pos_embedding = tAPE(d_model=cf_dim, dropout=dropout, max_len=5000, scale_factor=1.0)
        self.transformer = transformer(dim=cf_dim, depth=cf_depth, heads=heads,
                                       dim_head=cf_dim_head,
                                       mlp_dim=cf_mlp, dropout=dropout,
                                       use_att_rel=self.use_att_rel)
        self.mlp_head = nn.Linear(cf_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,patch_num):
        if self.use_pos:
            batch_patch,enc_in,patch_len = x.shape
            # print(x.shape,patch_num)
            batch_size = int(batch_patch / patch_num)
            x = rearrange(x, '(b p) n d -> (b n) p d',b=batch_size, p=patch_num,n=enc_in,d=patch_len)
            # x = x.reshape(batch_size,enc_in,patch_num,patch_len)
            # x = x.reshape(batch_size*enc_in,patch_num,patch_len)
            # print("patch_len",self.to_patch_embedding(x).shape)
            # x = self.pos_embedding(self.to_patch_embedding(x)).reshape(batch_size*patch_num,enc_in,-1)
            x = self.pos_embedding(self.to_patch_embedding(x))
            batch_vars, patch_num, patch_len = x.shape
            x = rearrange(x,
                          '(b n) p d -> (b p) n d',b=batch_size, p=patch_num,n=enc_in,d=patch_len)
            # print(x.shape)
        else:
            x = self.to_patch_embedding(x)
        x, attn = self.transformer(x)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x  # ,attn


#############
class transformer(nn.Module):  ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.2, use_att_rel=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.use_att_rel = use_att_rel
        if self.use_att_rel:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, rel_c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        # print("tranformer", x.shape)
        for attn, ff in self.layers:
            x_n, attn = attn(x)
            # print("x_n", x_n.shape)  # [B*patch_number,D,cff_dim]
            x = x_n + x
            x = ff(x) + x
        return x, attn

############




class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.2):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) / self.d_k

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn


class rel_c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.2):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # rel
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), dim_head))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

    def forward(self, x):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) / self.d_k
        attn = self.attend(dots)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn






#######################
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)