import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# x: [bs x nvars x d_model x patch_num]
# [bs * nvars x patch_num=12 x d_model]

class Graph_Encoder(nn.Module):
    def __init__(self,configs):
        super(Graph_Encoder, self).__init__()

        self.Graph_Layers = Graph_Layers(
            gnn=[GFC(configs.z, configs.z, configs.dropout, configs.d_model*2, seg=configs.seg) for i in
                 range(configs.graph_depth)],
            gl_layer=Graph_learning(configs.enc_in, configs.embed_dim, alpha=configs.gc_alpha),
            node_embs=nn.Parameter(torch.randn(configs.enc_in, configs.embed_dim), requires_grad=True),
            z=configs.z, d_model=configs.d_model*2, norm_layer=nn.LayerNorm(configs.d_model*2)
        )

    def forward(self,x):
         bs, nvars, patch_num, d_model,  = x.shape
         graph_x = rearrange(x, 'b n p d -> (b n) p d', b=bs, n=nvars,d=d_model,p=patch_num)
         enc_out = self.Graph_Layers(graph_x)
         enc_out = rearrange(enc_out, '(b n) p d -> b n d p', b=bs, n=nvars,d=d_model,p=patch_num)
         return enc_out




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
        # x [bs * nvars x patch_num x d_model]
        adj = self.graph_learning(self.node_embs)

        _,patch_ls,_ = x.shape
        g = rearrange(x, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])
        out = g.unsqueeze(1)
        out = self.scaler(out)

        for i, graph_net in enumerate(self.graph_net):

            out = graph_net(out, adj)

        out = self.group_concatenate(out).squeeze(-1)

        out = out.permute(0, 2, 1)
        g = rearrange(out, '(b p) n d -> (b n) p d', p=patch_ls)
        if self.norm is not None:
            x = self.norm(g)

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
        #print("GFC",x.shape,adj.shape)
        # GFC torch.Size([32, 32, 21, 512]) torch.Size([21, 21])
        #split
        if self.pad == 0:
            x = x.split([self.seg_dim] * self.seg, dim=1)
            #print("GFC_split", type(x),len(x),x[0].shape)
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
        out = torch.cat(out,dim=1)
        out = self.mlp(out)
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
        return adj





