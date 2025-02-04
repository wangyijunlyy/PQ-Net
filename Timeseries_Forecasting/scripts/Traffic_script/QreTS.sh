if [ ! -d "./logs/LongForecasting/traffic" ]; then
    mkdir ./logs/LongForecasting/traffic
fi

export CUDA_VISIBLE_DEVICES=3
datasets=traffic
ROOT_PATH="./dataset/traffic/"
DATA_PATH="traffic.csv"
MODEL="samPQN"
DATA="custom"
FEATURES=M
SEQ_LEN=512
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ENC_IN=862
DEC_IN=862
C_OUT=862
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
                --rho $rho \
                >logs/LongForecasting/$datasets/$MODEL'_'$MODEL_ID'_rho'$rho'_lr'$Learning_RATE.log 2>&1 
        done
    done
done
