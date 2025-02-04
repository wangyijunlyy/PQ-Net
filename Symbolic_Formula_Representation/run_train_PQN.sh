dataset_ids=(3)

for dataset_id in ${dataset_ids[@]}
do
    CUDA_VISIBLE_DEVICES=0 python train_PQN.py \
        --dataset_id $dataset_id \
        --dataset_dir dataset \
        --save_dir PQN_checkpoint

done