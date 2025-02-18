# Position the number of processes specified after the --nproc_per_node flag
torchrun --nproc_per_node 4 --master_port=25642 sft.py \
        --model_name base_model_path \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --dataset lastfm \
        --prompt_path ./prompt/music.txt \
        --logging_dir ./log/ \
        --output_dir xxx \
        --wandb_project xxx \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --eval_step 0.05 \
        --wandb_project wandb_proj_name \
        --wandb_name wandb_run_name
        