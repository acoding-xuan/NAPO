#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 新增变量定义，可以根据需要调整
neg_accept_ratio_values="0.7"
batch_shared_values="True"  # batch_shared 的两个值 True 和 False
neg_num_values="19 15 10 5 3 1"             # neg_num 参数的选择
dataset_values="lastfm"  # dataset 的选择，添加了 'movielens' 示例
prompt="music"
for gamma in 1.0
do
    for alpha1 in 0.5
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
                        torchrun --nproc_per_node 4 --master_port=25973 s-simpo-rec.py \
                            --dataset $dataset \
                            --model_name /NAS/liudx/LLM/llama_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
                            --resume_from_checkpoint ./output/${dataset}/sft/final_checkpoint \
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
                            --output_dir ./output/${dataset}/simpo_random/dataset_${dataset}_gamma_${gamma}_alpha1_${alpha1}_neg_accept_ratio_${neg_accept_ratio}_batch_shared_${batch_shared}_neg_num_${neg_num} \
                            --wandb_project simpo-ob-random \
                            --sample_neg_type random \
                            --neg_accept_ratio $neg_accept_ratio \
                            --wandb_name simpo-ob-random-dataset-${dataset}-gamma-${gamma}-alpha1-${alpha1}-neg_accept_ratio-${neg_accept_ratio}-batch_shared-${batch_shared}-neg_num-${neg_num} \
                            --sort_type seq_logits > simpo-random-dataset-${dataset}-gamma-${gamma}-alpha1-${alpha1}-neg_accept_ratio-${neg_accept_ratio}-batch_shared-${batch_shared}-neg_num-${neg_num}.log
                    done
                done
            done
        done
    done
done
