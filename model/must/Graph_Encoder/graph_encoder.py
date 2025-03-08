import torch
import torch.nn as nn
import torch.nn.functional as F
from model.must.Graph_Encoder.Embed import PatchEmbedding
from einops import rearrange

class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        self.nf = nf

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        print("nf",self.nf,x.shape)
        x = self.linear(x)
        x = self.dropout(x)
        # nf 6144 torch.Size([32, 21, 6144])
        return x
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        # x = self.conv(x)
        return x

class GFC(nn.Module):
    def __init__(self, c_in, c_out,dropout,d_model,seg=4):
        super(GFC, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in , c_out)
        self.dropout = dropout
        self.alpha = 0.5
        self.seg = seg
        self.seg_dim = c_in // self.seg
        self.pad = c_in % self.seg
        self.agg = nn.ModuleList()
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,3], stride=1, padding=[0,1]))
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,5], stride=1, padding=[0,2]))
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,7], stride=1, padding=[0,3]))

        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        #adj
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        a = adj / d.view(-1, 1)
        print("GFC",x.shape,adj.shape)
        # GFC torch.Size([32, 32, 21, 512]) torch.Size([21, 21])
        #split
        if self.pad == 0:
            x = x.split([self.seg_dim] * self.seg, dim=1)
            print("GFC_split", type(x),len(x),x[0].shape)
            # GFC_split <class 'tuple'> 4 torch.Size([32, 8=32//4, 21, 512])
            out = [x[0]]
            # (B, c, N ,d_model)
            for i in range(1, self.seg):
                h = self.agg[i-1](x[i])
                h = self.alpha * (h + x[i]) + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)
        else:
            y = x[:, :self.seg_dim + self.pad, :, :]
            out = [y]
            x = x[:, self.seg_dim + self.pad:, :, :]
            x = x.split([self.seg_dim] * (self.seg-1),dim=1)
            # (B, c, N ,d_model)
            for i in range(0, self.seg-1):
                h = self.agg[i](x[i])
                h = self.alpha * (h + x[i]) + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)
        print("out_list",out[0].shape)
        # # GFC_split <class 'tuple'> 4 torch.Size([32, 8, 21, 512])
        out = torch.cat(out,dim=1)
        print("GFC_out",out.shape)
        # # GFC_out torch.Size([32, 32, 21, 512])
        out = self.mlp(out)
        print("_out", out.shape)
        out_1 = self.gelu(out)

        return  self.norm(out_1) + out


class Graph_learning(nn.Module):
    def __init__(self, nnodes, emb_dim, alpha=1):

        """
        :param nnodes:  enc_in,node_number
        :param emb_dim: Ada embedding dimension
        :param alpha:   vec_1,2
        input：
        """
        super(Graph_learning, self).__init__()
        self.nnodes = nnodes

        self.lin1 = nn.Linear(emb_dim,emb_dim)
        self.lin2 = nn.Linear(emb_dim,emb_dim)

        self.emb_dim = emb_dim
        self.alpha = alpha

    def forward(self, node_emb):
        # node_emb:[nnodes,emb_dim]
        nodevec1 = F.gelu(self.alpha*self.lin1(node_emb))
        nodevec2 = F.gelu(self.alpha*self.lin2(node_emb))
        adj = F.relu(torch.mm(nodevec1, nodevec2.transpose(1,0)) - torch.mm(nodevec2, nodevec1.transpose(1,0)))
        print("adj",adj.shape)
        # adj torch.Size([21, 21])
        return adj



class Graph_Layers(nn.Module):
    def __init__(self, gnn, gl_layer, node_embs,z,d_model, norm_layer=None):
        super(Graph_Layers, self).__init__()

        self.graph_net = nn.ModuleList(gnn)
        self.graph_learning = gl_layer
        self.norm = norm_layer
        self.node_embs = node_embs

        self.scaler = nn.Conv2d(1, z, (1, 1))
        self.group_concatenate = nn.Conv2d(z, d_model, (1, d_model))


    def forward(self,x):
        # x [bs * nvars x patch_num+global_len x d_model]
        print("xxx",x.shape)
        adj = self.graph_learning(self.node_embs)

        _,patch_ls,_ = x.shape
        g = rearrange(x, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])
        out = g.unsqueeze(1)
        out = self.scaler(out)

        for i, graph_net in enumerate(self.graph_net):

            out = graph_net(out, adj)

        out = self.group_concatenate(out).squeeze(-1)

        print("out",out.shape)
        out = out.permute(0, 2, 1)
        print("out11",out.shape)
        g = rearrange(out, '(b p) n d -> (b n) p d', p=patch_ls)
        print("g",g.shape,x.shape,x.shape)
        x = g
        if self.norm is not None:
            x = self.norm(x)

        return x


class GraphEncoder(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8, gc_alpha=1):
        super(GraphEncoder, self).__init__()

        # padding
        padding = configs.stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        # Graph_Layers
        self.Graph_Layers = Graph_Layers(
            gnn=[GFC(configs.z, configs.z, configs.dropout,configs.d_model, seg=configs.seg) for i in range(configs.graph_depth)],
            gl_layer=Graph_learning(configs.enc_in, configs.embed_dim, alpha=configs.gc_alpha),
            node_embs=nn.Parameter(torch.randn(configs.enc_in, configs.embed_dim), requires_grad=True),
            z=configs.z, d_model=configs.d_model, norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.head = Flatten_Head(configs.enc_in, self.head_nf, configs.seq_len,
                                 head_dropout=configs.dropout)
    def forward(self,x_enc):
        """
        x_in [batch_size,seq_len,enc_in]
        """
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs * nvars x patch_num=12 x d_model]

        print("enc_out", enc_out.shape)
        # enc_out torch.Size([672, 15, 512])

        # Encoder
        # z: [bs * nvars x patch_num+global_len x d_model]

        enc_out = self.Graph_Layers(enc_out)
        enc_out = enc_out
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        enc_out = enc_out.permute(0, 1, 3, 2)
        # gg torch.Size([32, 21, 512, 12])

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)  #[32, 21, 6144]

        return dec_out

if __name__ == "__main__":
    from config import parse_args
    configs = parse_args()
    model = GraphEncoder(configs)
    x = torch.randn((256,96,11))
    output = model(x)
    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    print(output.shape)



