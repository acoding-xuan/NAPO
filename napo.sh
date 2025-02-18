#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
neg_accept_ratio_values="0.7"
batch_shared_values="True"  
neg_num_values="3"            
dataset_values="lastfm"  
prompt="music"
for gamma in 1.0
do
    for alpha1 in 0.3
    do
        for neg_accept_ratio in $neg_accept_ratio_values
        do
            for batch_shared in $batch_shared_values
            do
                for neg_num in $neg_num_values
                do
                    for dataset in $dataset_values
                    do
                        echo "Running with dataset: $dataset, gamma: $gamma, alpha1: $alpha1, neg_accept_ratio: $neg_accept_ratio, batch_shared: $batch_shared, neg_num: $neg_num"
                        torchrun --nproc_per_node 4 --master_port=25971 napo.py \
                            --dataset $dataset \
                            --model_name base_model_path \
                            --resume_from_checkpoint ref_model_ckpt_path \
                            --batch_size 4 \
                            --gradient_accumulation_steps 8 \
                            --prompt_path ./prompt/${prompt}.txt \
                            --learning_rate 1e-5 \
                            --eval_step 0.033 \
                            --beta 1 \
                            --gamma $gamma \
                            --alpha1 $alpha1 \
                            --move_ratio 0.9 \
                            --neg_num $neg_num \
                            --batch_shared $batch_shared \
                            --num_train_epochs 3 \
                            --logging_dir ./log/ \
                            --output_dir ./output/${dataset}/ \
                            --wandb_project xxx \
                            --sample_neg_type random \
                            --neg_accept_ratio $neg_accept_ratio \
                            --wandb_name xxx \
                            --sort_type seq_logits 
                    done
                done
            done
        done
    done
done
