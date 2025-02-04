dataset_ids=(3)

for dataset_id in ${dataset_ids[@]}
do
    CUDA_VISIBLE_DEVICES=1 python train_mlp.py \
        --dataset_id $dataset_id \
        --dataset_dir dataset \
        --save_dir mlp_checkpoint
        
done