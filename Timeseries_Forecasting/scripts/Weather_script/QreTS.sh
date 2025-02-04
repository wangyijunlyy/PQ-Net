if [ ! -d "./logs/LongForecasting/weather" ]; then
    mkdir ./logs/LongForecasting/weather
fi

export CUDA_VISIBLE_DEVICES=5
datasets=weather
ROOT_PATH="./dataset/weather/"
DATA_PATH="weather.csv"
MODEL="QreLinear"
DATA="custom"
FEATURES=M
SEQ_LEN=512
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ENC_IN=21
DEC_IN=21
C_OUT=21
DES="Exp"
ITR=1


# 96 192 336 720
for PRED_LEN in 96 192 336 720
do
    for rho in 0.7
    do
        for Learning_RATE in 1e-3
        do
            MODEL_ID="${datasets}_${SEQ_LEN}_${PRED_LEN}"
            
            
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u run.py \
                --is_training 1 \
                --root_path $ROOT_PATH \
                --data_path $DATA_PATH \
                --model_id $MODEL_ID \
                --model $MODEL \
                --data $DATA \
                --features $FEATURES \
                --seq_len $SEQ_LEN \
                --label_len $LABEL_LEN \
                --pred_len $PRED_LEN \
                --e_layers $E_LAYERS \
                --d_layers $D_LAYERS \
                --factor $FACTOR \
                --enc_in $ENC_IN \
                --dec_in $DEC_IN \
                --c_out $C_OUT \
                --des $DES \
                --itr $ITR \
                --learning_rate $Learning_RATE \
                >logs/LongForecasting/$datasets/$MODEL'_'$MODEL_ID'_lr'$Learning_RATE.log 2>&1 
        done
    done
done
