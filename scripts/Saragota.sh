seed=2023
mask=dozer_ext_0
lr=5e-5
model=dozerformer_Linear
patch_size=60
batch_size=24
# Exformer attention parameters
local_window=2
stride=4
vary_len=1
train=1
fusion=SUM
anorm_thres=0.8

gpu=1
devices=1

patch_thres=1
pred_len=288

echo "=============================================="
echo "Running Seed: $seed | patch_size: $patch_size"
echo "=============================================="
echo "Prediction Length: $pred_len"

python run.py \
  --seed $seed \
  --batch_size $batch_size \
  --data Saratoga \
  --is_training $train \
  --model $model \
  --moving_avg '13, 17' \
  --seq_len 1440 \
  --label_len 96 \
  --pred_len $pred_len \
  --embed_dim 15 \
  --learning_rate $lr \
  --patch_size $patch_size \
  --local_window $local_window \
  --stride $stride \
  --vary_len $vary_len \
  --mask $mask \
  --patch_thres $patch_thres \
  --gpu $gpu \
  --devices $devices \
  --fusion $fusion \
  --anorm_thres $anorm_thres

printf '\a'