import argparse

parser = argparse.ArgumentParser()
def parserss():
#Fredformer:
    parser.add_argument('--cf_dim', type=int, default=48)   #feature dimension
    parser.add_argument('--cf_depth',type=int, default=2)    #Transformer layer
    parser.add_argument('--cf_heads',type=int, default=6)    #number of multi-heads
    #parser.add_argument('--cf_patch_len',  type=int, default=16)   #patch length
    parser.add_argument('--cf_mlp', type=int, default=128)  #ff dimension
    parser.add_argument('--cf_head_dim',type=int, default=32)   #dimension for single head


    parser.add_argument('--use_pos',type=bool, default=True)    #ablation study 012.
    parser.add_argument('--use_att_rel',type=bool, default=True)    #ablation study 012.


    parser.add_argument('--d_model',type=int, default=128)    #ablation study 012.
    parser.add_argument('--seq_len',type=int, default=96)    #ablation study 012.
    parser.add_argument('--patch_len',type=int, default=16)    #ablation study 012.
    parser.add_argument('--stride',type=int, default=8)    #ablation study 012.
    parser.add_argument('--dropout',type=int, default=0.2)    #ablation study 012.
    parser.add_argument('--enc_in',type=int, default=9)    #ablation study 012.
    parser.add_argument('--pred_len',type=int, default=24)    #ablation study 012.

    args = parser.parse_args()
    return args
