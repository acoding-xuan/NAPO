# Position the number of processes specified after the --nproc_per_node flag
torchrun --nproc_per_node 4 --master_port=25642 sft.py \
        --model_name /NAS/liudx/LLM/llama_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --dataset lastfm \
        --prompt_path ./prompt/music.txt \
        --logging_dir ./log/ \
        --output_dir ./output/lastfm/sft/ \
        --wandb_project dpo-rec-nf4 \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --eval_step 0.05 \
        --wandb_project wandb_proj_name \
        --wandb_name wandb_run_name > sft.log
        