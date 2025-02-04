if [ ! -d "./logs/LongForecasting/exchange_rate" ]; then
    mkdir ./logs/LongForecasting/exchange_rate
fi

export CUDA_VISIBLE_DEVICES=3
datasets=exchange_rate
ROOT_PATH="./dataset/exchange_rate/"
DATA_PATH="exchange_rate.csv"
MODEL="samPQN"
DATA="custom"
FEATURES=M
SEQ_LEN=512
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ENC_IN=8
DEC_IN=8
C_OUT=8
DES="Exp"
ITR=1
Learning_RATE=1e-3

# 96 192 336 720
for PRED_LEN in 96 192 336 720
do
    for rho in 0.7 0.9
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
            --rho $rho \
            >logs/LongForecasting/$datasets/$MODEL'_'$MODEL_ID'_rho'$rho.log 2>&1 
    done
done
