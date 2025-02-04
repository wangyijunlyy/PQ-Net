export CUDA_VISIBLE_DEVICES=2

train_epochs=50

for pred_len in 96 192 336 720
do
  model_id="weather_96_${pred_len}"
  
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
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
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs $train_epochs >logs/LongForecasting/weather/baseline_$model_id.log 2>&1
done
