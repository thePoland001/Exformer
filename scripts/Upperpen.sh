# Fixed Parameters
model=dozerformer_Linear
lr=5e-5
batch_size=24
local_window=2
stride=4
vary_len=1
train=1
fusion=SUM
dataset=UpperPen
gpu=0
devices=0
seq_len=1440
label_len=96
embed_dim=15
moving_avg='13, 17'

seed=2023
mask=dozer
patch_size=60
patch_thres=1
pred_len=288
anorm_thres=0.77

# Run Experiment
echo "============================================="
echo "Seed: $seed | Mask: $mask |  Pred: $pred_len"
echo "=============================================="

python run.py \
  --seed "$seed" \
  --batch_size "$batch_size" \
  --data "$dataset" \
  --is_training "$train" \
  --model "$model" \
  --moving_avg "$moving_avg" \
  --seq_len "$seq_len" \
  --label_len "$label_len" \
  --pred_len "$pred_len" \
  --embed_dim "$embed_dim" \
  --learning_rate "$lr" \
  --patch_size "$patch_size" \
  --local_window "$local_window" \
  --stride "$stride" \
  --vary_len "$vary_len" \
  --mask "$mask" \
  --patch_thres "$patch_thres" \
  --gpu "$gpu" \
  --devices "$devices" \
  --fusion "$fusion" \
  --anorm_thres "$anorm_thres"

printf '\a'