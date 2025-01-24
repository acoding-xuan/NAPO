#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# 定义所有的检查点路径
checkpoints=(
#     "./output_1/lastfm/high_gamma_0.0_final_checkpoint"
#     "./output_1/lastfm/high_gamma_0.5_final_checkpoint"
#     "./output_1/lastfm/high_gamma_1.0_final_checkpoint"
#     "./output_1/lastfm/high_gamma_1.5_final_checkpoint"
#     "./output_1/lastfm/low_gamma_0.0_final_checkpoint"
#     "./output_1/lastfm/low_gamma_0.5_final_checkpoint"
#     "./output_1/lastfm/low_gamma_1.0_final_checkpoint"
#     "./output_1/lastfm/low_gamma_1.5_final_checkpoint"
#     "./output_1/lastfm/low_gamma_2.0_final_checkpoint"
#     "./output_1/lastfm/random_gamma_0.0_final_checkpoint"
#     "./output_1/lastfm/random_gamma_0.5_final_checkpoint"
#     "./output_1/lastfm/random_gamma_1.0_final_checkpoint"
    # "./output/lastfm/random_dyg_gamma_0.0_final_checkpoint"
    # "./output/lastfm/neg_random_dyg_gamma_1.0_final_checkpoint"
    #"./output/goodreads/goodreads_random_dyg_gamma_1.0_alpha1_0.1_neg_accept_ratio_0.7_batch_shared_True_neg_num_3_final_checkpoint"
    "./output/goodreads/goodreads_random_dyg_gamma_1.0_alpha1_0.3_neg_accept_ratio_0.7_batch_shared_True_neg_num_3_2gpu_final_checkpoint"
 #  "./output/lastfm/random_dyg_gamma_1.5_alpha1_0.3_neg_accept_ratio_0.7_discard_ratio_0.0_final_checkpoint"
)

# 循环遍历每个检查点
for checkpoint in "${checkpoints[@]}"
do
    # 提取文件名的一部分作为日志文件名 basename 是一个命令，用于去掉路径中的目录部分，只返回文件名。
    filename=$(basename "$checkpoint")
    
    echo "Running inference with checkpoint: $checkpoint"

    # 运行 torchrun 命令
    torchrun --nproc_per_node 1 --master_port=25647 \
        inference.py \
        --dataset goodreads \
        --external_prompt_path ./prompt/book.txt \
        --batch_size 64 \
        --base_model /NAS/liudx/LLM/llama_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
        --resume_from_checkpoint "$checkpoint" \
        > "eval_$filename.log"
done