# Position the number of processes specified after the --nproc_per_node flag
torchrun --nproc_per_node 4 --master_port=25643 softmax_dpo.py \
            --model_name /data/liudx/LLM/llama_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
            --resume_from_checkpoint /data/liudx/S-DPO/output/lastfm/final_checkpoint  \
            --batch_size 4 \
            --gradient_accumulation_steps 8 \
            --dataset lastfm \
            --prompt_path ./prompt/music.txt \
            --learning_rate 1e-5 \
            --eval_step 0.033 \
            --beta 1 \
            --neg_num 1 \
            --num_train_epochs 3 \
            --logging_dir ./log/\
            --output_dir ./output/lastfm/dpo/ \
            --wandb_project sdpo-dpo \
            --wandb_name sdpo-dpo-1 > dpo.log