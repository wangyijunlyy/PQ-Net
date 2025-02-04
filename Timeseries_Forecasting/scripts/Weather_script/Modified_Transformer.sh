export CUDA_VISIBLE_DEVICES=1

datasets=("weather") 
pred_lens=(96 192 336 720)
model="Modified_Transformer"
exp_setting=5
learning_rate=1e-4

for dataset in "${datasets[@]}"; do
  if [ ! -d "./logs/LongForecasting/$dataset" ]; then
    mkdir ./logs/LongForecasting/$dataset
  fi
  for pred_len in "${pred_lens[@]}"; do
    model_id="${dataset}_96_${pred_len}"
    echo "Running model: $model_id"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u run.py \
      --is_training 1 \
      --root_path "./dataset/weather/" \
      --data_path "${dataset}.csv" \
      --model_id "$model_id" \
      --model "$model" \
      --data "custom" \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "$pred_len" \
      --e_layers 2 \
      --d_layers 1 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --freq 't' \
      --learning_rate $learning_rate \
      --exp_setting $exp_setting \
      --itr 1 >logs/LongForecasting/$dataset/$model'_'$model_id'_'exp_setting_$exp_setting'_'lr$learning_rate.log 2>&1
  done
done
