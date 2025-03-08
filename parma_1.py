import argparse
import json
import torch

def parse_args():
########################################################################3
    parser = argparse.ArgumentParser(description="Hierarchical Graph Forecasting Experiment ")
    parser.add_argument('--hier_method', type=str, default='ST_htqf', help="ST_htqf,ST_htqf_hier,Must,Soft,")
    parser.add_argument('--method', type=str, default='XX', help="no mean")
    parser.add_argument('--methods', type=str, default='BU', help="hierarchical_forecasting_method")
    parser.add_argument('--hier_number', type=int, default=15, help='NSW1:15,SA1:25')

    parser.add_argument('--root_path', type=str, default='./io/input/', help='root_path')
    parser.add_argument('--data_path', type=str, default='NSW1_data.csv', help='data_path')
    parser.add_argument('--data_S_path', type=str, default='NSW1_S.npy', help='data_path')
    parser.add_argument('--data_G_path', type=str, default='NSW1_G.npy', help='data_path')
    parser.add_argument('--site', type=str, default='NSW1', help="data_site")

    # NCG、Cross、NCross
    parser.add_argument('--model', type=str, default="Graph_Freq", help='model_name')
    parser.add_argument('--data_dim', type=int, default=9, help='NSW1:9,SA1:20)')
    parser.add_argument('--enc_in', type=int, default=9, help='enc_in parameter.15,25')
    parser.add_argument('--dec_in', type=int, default=9, help='dec_in parameter')
    parser.add_argument('--c_out', type=int, default=9, help='c_out parameter')
    parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
    parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')

    # Transformer
    parser.add_argument('--seq_len', type=int, default=96, help='Sequence length')
    parser.add_argument('--label_len', type=int, default=32, help='Label length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction length')
    parser.add_argument('--in_len', type=int, default=96, help='input_len')
    parser.add_argument('--out_len', type=int, default=24, help='output_len')

    # Graph_Encoder
    parser.add_argument('--embed_dim', type=int, default=10, help='ada graph embedding dimension')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--patch_len', type=int, default=16, help='patch_len')
    parser.add_argument('--gc_alpha', type=int, default=0.5, help='da graph_alpha')
    parser.add_argument('--z', type=int, default=32, help='scaler_z')
    parser.add_argument('--seg', type=int, default=4, help='seg')
    parser.add_argument('--Multivariate_type',type=str, default="Graph")
    parser.add_argument('--graph_depth', type=int, default=2, help='graph_depth')

    #Fredformer:
    parser.add_argument('--cf_dim', type=int, default=48)
    parser.add_argument('--cf_depth',type=int, default=2)
    parser.add_argument('--cf_heads',type=int, default=6)
    parser.add_argument('--cf_patch_len',  type=int, default=16)
    parser.add_argument('--cf_mlp', type=int, default=128)
    parser.add_argument('--cf_head_dim',type=int, default=32)
    parser.add_argument('--use_pos',type=bool, default=True)
    parser.add_argument('--use_att_rel',type=bool, default=True)

    # Autoformer
    parser.add_argument('--d_model', type=int, default=128, help='d_model parameter')
    parser.add_argument('--d_ff', type=int, default=512, help='d_ff parameter')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--factor', type=int, default=5,help='Factor parameter')

    parser.add_argument('--moving_avg', type=int, default=25, help='Moving average parameter')
    parser.add_argument('--version', type=str, default='Fourier', help='Version parameter')
    parser.add_argument('--mode_select', type=str, default='random', help='Mode selection parameter')
    parser.add_argument('--modes', type=int, default=64, help='Number of modes')
    parser.add_argument('--output_attention', action='store_true', help='Output attention flag')
    parser.add_argument('--freq', type=str, default='h', help='Frequency parameter')
    parser.add_argument('--embed_type', type=int, default=0, help='Embedding type parameter')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--embed', type=str, default='timeF', help='Embedding parameter')

    # iTransformer
    parser.add_argument('--depth', type=int, default=6, help='depth parameter')
    parser.add_argument('--dim_head', type=int, default=48, help='head_dim parameter')

    # triformer
    parser.add_argument('--mem_dim', type=int, default=32, help='men_dim')

    # Data
    parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2', help='data split of train, vali, test')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers,none in this exp')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=6, help='Early Stopping')

    # cuda
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--seed', type=int, default=3047, help='lucky seed')


    # q
    parser.add_argument('--output_dim', type=int, default=4, help='htqf:4')
    parser.add_argument('--loss_fun', type=str, default='htqf', help='htqf,mse,htqf_hier')

    return parser.parse_args()


def get_setting(config):
    setting = '{}/{}/{}+{}/{}_{}_{}_ed{}_hd{}i{}_if{}_o{}_s{}_d{}_nh{}_el{}_df{}_f{}_dr{}_bs{}_ep{}_lr{}'.format(
        config.site,
        config.loss_fun,
        config.seq_len,
        config.pred_len,
        config.model,
        config.hier_method,
        config.method,
        config.data_dim,
        config.hier_number,
        config.seq_len,
        config.label_len,
        config.pred_len,
        config.seg_len,
        config.d_model,
        config.n_heads,
        config.e_layers,
        config.d_ff,
        config.factor,
        config.dropout,
        config.batch_size,
        config.epochs,
        config.lr)
    return setting


