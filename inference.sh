# The number of processes can only be one for inference
#torchrun --nproc_per_node 1 --master_port=25642 \
torchrun --nproc_per_node 1 --master_port=25688 \
        inference.py \
        --dataset lastfm \
        --external_prompt_path ./prompt/music.txt \
        --batch_size 32 \
        --base_model /NAS/liudx/LLM/llama_model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
        --resume_from_checkpoint xxx \