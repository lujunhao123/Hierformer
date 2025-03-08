import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ST_htqf_hier.Graph_Freq.Freq_tAPE_rel.tAPE_rel_cross_transformer import Cross_Transformer
class Impoveformer(nn.Module):
    def __init__(self,config):
        super(Impoveformer, self).__init__()

        self.get_r = nn.Linear(config.d_model*2, config.d_model*2)
        self.get_i = nn.Linear(config.d_model*2, config.d_model*2)
        self.ircom = nn.Linear(config.pred_len * 2, config.pred_len)

        self.patch_len = config.patch_len
        self.stride = config.stride
        patch_num = int((config.seq_len - config.patch_len) / config.stride + 1)
        self.head_nf_f = config.d_model * 2 * patch_num

        self.head_f1 = Flatten_Head(individual=True,
                                         n_vars=config.enc_in, nf=self.head_nf_f,
                                         target_window=config.pred_len,
                                         head_dropout=0)
        self.head_f2 = Flatten_Head(individual=True,
                                         n_vars=config.enc_in, nf=self.head_nf_f,
                                         target_window=config.pred_len,
                                         head_dropout=0)

        self.cross_transformer = Cross_Transformer(
            patch_dim=config.patch_len*2,patch_num=patch_num,
            cf_dim=config.cf_dim, cf_mlp=config.cf_mlp,cf_depth=config.cf_depth, heads=config.cf_heads,
            cf_dim_head=config.cf_head_dim, d_model=config.d_model*2, dropout=config.dropout,
            use_pos=config.use_pos, use_att_rel=config.use_att_rel
        )


    def forward(self,x):
        seq_x = x.permute(0, 2, 1)  # [B,D,S]

        z = torch.fft.fft(seq_x)
        z1 = z.real
        z2 = z.imag
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_len = z1.shape[3]

        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))  # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))  # z: [bs * patch_num,nvars, patch_len]
        # print("freqformer_input", torch.cat((z1, z2), -1).shape)
        # [B*patch_number,input_dim,patch_dim[real+image]]
        z = self.cross_transformer(torch.cat((z1, z2), -1),patch_num)  ##[B*patch_number.input_dim,d_model*2]
        # print("model_output",z.shape)
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        ###########
        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        # print("head_f1_input", z1.shape, z2.shape)  # [bs, nvars， patch_num, d_model*2]

        z1 = self.head_f1(z1)  # z: [bs x nvars x target_window]
        z2 = self.head_f2(z2)  # z: [bs x nvars x target_window]
        # print("head_f1_output", z1.shape)
        # print("ifft_input", torch.complex(z1, z2).shape)
        #exit()
        z = torch.fft.ifft(torch.complex(z1, z2))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))
        z = z.permute(0, 2, 1)

        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            # self.linears2 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                # self.linears2.append(nn.Linear(target_window, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)  # z: [bs x target_window]
                # z = self.linears2[i](z)                    # z: [target_window x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
            # x = self.linear1(x)
            # x = self.linear2(x) + x
            # x = self.dropout(x)
        return x


if __name__ == '__main__':
    x = torch.randn((32,96,9))
    from pars import parserss
    config = parserss()
    model = Impoveformer(config)
    output = model(x)
    print(output.shape)