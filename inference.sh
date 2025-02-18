# The number of processes can only be one for inference
#torchrun --nproc_per_node 1 --master_port=25642 \
torchrun --nproc_per_node 1 --master_port=25688 \
        inference.py \
        --dataset lastfm \
        --external_prompt_path ./prompt/music.txt \
        --batch_size 32 \
        --base_model base_model \
        --resume_from_checkpoint ckpt_path \