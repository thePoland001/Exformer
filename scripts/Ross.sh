seed=2023
mask=dozer_ext_0
attn=FedAttn

lr=5e-5
model=dozerformer_Linear
patch_size=12
batch_size=24
# Exformer attention parameters
local_window=2
stride=4
vary_len=1
train=1
fusion=SUM
data=Ross
anorm_thre=0.9
patch_thres=1
gpu=0
devices=0
pred_len=288

echo "================================================================"
echo "Running Seed: $seed | Mask: $mask | "
echo "================================================================"
echo "Prediction Length: $pred_len"

python run.py \
  --seed $seed \
  --exp_run $exp \
  --batch_size $batch_size \
  --data $data \
  --is_training $train \
  --model $model \
  --moving_avg '13, 17' \
  --seq_len 1440 \
  --label_len 96 \
  --pred_len $pred_len \
  --embed_dim 8 \
  --learning_rate $lr \
  --patch_size $patch_size \
  --local_window $local_window \
  --stride $stride \
  --vary_len $vary_len \
  --mask $mask \
  --patch_thres $patch_thres \
  --fusion $fusion \
  --gpu $gpu \
  --devices $devices \
  --anorm_thres $anorm_thre \
  --attn $attn

printf '\a'
