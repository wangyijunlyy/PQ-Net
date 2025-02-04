export CUDA_VISIBLE_DEVICES=3

train_epochs=10

for pred_len in 96 192 336 720
do
  model_id="traffic_96_${pred_len}"
  
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id $model_id \
    --model Transformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs $train_epochs >logs/LongForecasting/traffic/baseline_$model_id.log 2>&1
done
