#!/bin/bash

for model in Autoformer Crossformer Informer GRU TCN LSTM
do

for seq_len in 96 48
do

for pred_len_del in 4 2
do

######## ST_hier #########
#  NSW1
python -u Wind_ST_hier_q.py \
  --hier_method ST_hier \
  --loss_fun mse \
  --model $model \
  --site NSW1\
  --data_path NSW1_data.csv \
  --data_S_path NSW1_S.npy \
  --data_G_path NSW1_G.npy \
  --method XX \
  --in_len $seq_len \
  --out_len $((seq_len / pred_len_del)) \
  --seq_len $seq_len \
  --label_len $((seq_len / pred_len_del)) \
  --pred_len $((seq_len / pred_len_del)) \
  --patch_len $((seq_len / 6)) \
  --stride $((seq_len / 12)) \
  --data_dim 9 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --hier_number 15 \
  --embed_dim 18 \
  --d_model 128 \
  --batch_size 128 \
  --n_heads 8 \
  --factor 3 \
  --epochs 20 \
  --cf_dim 64 \
  --cf_depth 2 \
  --cf_heads 6 \
  --cf_head_dim 32 \
  --device cuda \
  --seed 3047

######## ST_hier #########
#  SA1
python -u Wind_ST_hier_q.py \
  --hier_method ST_hier \
  --loss_fun mse \
  --model $model \
  --site SA1\
  --data_path SA1_data.csv \
  --data_S_path SA1_S.npy \
  --data_G_path SA1_G.npy \
  --method XX\
  --in_len $seq_len \
  --out_len $((seq_len / pred_len_del)) \
  --seq_len $seq_len \
  --label_len $((seq_len / pred_len_del)) \
  --pred_len $((seq_len / pred_len_del)) \
  --patch_len $((seq_len / 6)) \
  --stride $((seq_len / 12)) \
  --data_dim 20 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --hier_number 25 \
  --embed_dim 40 \
  --d_model 256 \
  --batch_size 128 \
  --n_heads 8 \
  --factor 3 \
  --epochs 20 \
  --cf_dim 128 \
  --cf_depth 2 \
  --cf_heads 8 \
  --cf_head_dim 64 \
  --device cuda \
  --seed 3047


done
done
done