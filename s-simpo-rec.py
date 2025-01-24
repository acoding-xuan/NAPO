import sklearn
import os
import transformers
from recommender.A_SASRec_final_bce_llm import SASRec
import torch
import re
import random
import numpy as np

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
# from trl import DPOTrainer
from trainer.softmax_dpo_trainer import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainerCallback
#from data.lastfm_data import LastfmData

from Prompt import Prompt

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire

random.seed(1958)

def train(
    #train
    output_dir="",
    logging_dir="",
    model_name ="",
    prompt_path = "",
    dataset="",
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    alpha1: float = 0.3,
    alpha2: float = 0.3,
    beta: float = 0.1,
    gamma: float = 1.0,
    neg_num: int = 3,
    move_ratio: float = 0.9,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step = 5,  
    sort_type = "seq_logits", 
    neg_accept_ratio = 0.7, 
    sample_neg_type = "high",
    batch_shared=True,
):
    #os.environ['WANDB_PROJECT'] = wandb_project # 指定wandb 项目名称
    data_files = { 
        "train": f"./data/{dataset}-sft-cans20/{dataset}-train.json",
        "validation": f"./data/{dataset}-sft-cans20/{dataset}-val.json",
    }
    print(sample_neg_type)
    print("gamma : ", gamma)
    print("alpha1 : ", alpha1)
    print("sort_type : ", sort_type)
    print("neg_accept_ratio : ", neg_accept_ratio)
    print("move_ratio : ", move_ratio)
    print("beta : ", beta)
    print("neg_num : ", neg_num)
    print("batch_shared : ", batch_shared)
    print("cutoff_len", cutoff_len)
    def convert_dict_to_prompt(d:dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t
    
    def process_data(examples):
        dic = {"prompt":[], "chosen":[], "dataset_id":[]}
        for i in range(1, neg_num+1):
            dic[f"rejected{i}"] = []
            dic[f"score_rejected{i}"] = []
        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {}
            data_point["trueSelection"] = examples["trueSelection"][i]
            data_point["itemList"] = examples["itemList"][i]
            data_point["historyList"] = examples["historyList"][i]   
            sorted_neg_items = examples["sorted_neg_items"][i] 
            sorted_neg_items_score = examples["scores_of_sorted_neg_items"][i]
            t = convert_dict_to_prompt(data_point)
            prompt = str(t)
            chosen = data_point["trueSelection"]
            negative_items = [item for item in data_point["itemList"] if item != data_point["trueSelection"]]
            sample_negs = random.sample( negative_items, neg_num)
            indices = [sorted_neg_items.index(item) for item in sample_negs]
            sample_negs_score = [sorted_neg_items_score[indice] for indice in indices]
            dic["prompt"].append(prompt)
            dic["dataset_id"].append(examples["dataset_id"][i])
            dic["chosen"].append(chosen)
            cnt = 0  
            for rejected, reject_score in zip(sample_negs, sample_negs_score):
                cnt += 1
                dic[f"rejected{cnt}"].append(rejected)
                dic[f"score_rejected{cnt}"].append(reject_score)   
        return dic

    
    def model_init():
        device_index = Accelerator().process_index # 创建一个 Accelerator 对象，并调用其 process_index 属性来获取当前进程的索引。
        device_map = {"": device_index}
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                    device_map=device_map, 
                                                    # load_in_8bit=True,
                                                    # torch_dtype=torch.bfloat16,
                                                    quantization_config=bnb_config)
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model)
        base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, 
                                            is_trainable=True)
        # print_trainable_parameters(base_model)
        #base_model.print_trainable_parameters()
        return base_model
        

    data = load_dataset("json", data_files=data_files)

    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, \
                                    num_proc=8, batched=True).shuffle(seed=42)
    print(train_data)

    # random 2000 samples for validation
    val_data = data["validation"].map(process_data, remove_columns=columns, \
                                        num_proc=8, batched=True).shuffle(seed=42)
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))
    
    print(val_data)
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps", 
        save_steps=eval_step * 10,
        save_total_limit=3,
        evaluation_strategy="steps", # `"steps"`: Evaluation is done (and logged) every `eval_steps`.
        eval_steps=eval_step,
        load_best_model_at_end=True,
        logging_steps=1,
        output_dir=output_dir,
        report_to = None,
        #run_name = wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False,
    )
    #rec_model_path = f"./rec_model/lastfm.pt"
    device_index = Accelerator().process_index 
    torch.cuda.set_device(device_index)
    #sasrec_model = torch.load(rec_model_path).cuda()
    base_model = model_init() 

    item_embeddings = torch.tensor(data['train']['avg_item_embedding'], device='cuda')
    seq_logits = torch.tensor(data['train']['seq_logits'], device='cuda')

    def batch_calculate_similarity(ids1, ids2, sort_type="item_embedding"):
        if sort_type == "item_embedding":
            emb1 = item_embeddings[ids1]
            emb2 = item_embeddings[ids2]
        elif sort_type == "seq_logits":
            emb1 = seq_logits[ids1]
            emb2 = seq_logits[ids2]
        # 计算批量的点积
        return torch.matmul(emb1, emb2.T)  # 结果是一个矩阵，表示所有向量之间的点积

    def calculate_similarity(ids1, ids2):
        if sort_type == "item_embedding": 
            return batch_calculate_similarity(ids1, ids2, sort_type="item_embedding")
        elif sort_type == "seq_logits":
            return batch_calculate_similarity(ids1, ids2, sort_type="seq_logits")
        
    similarity_score = calculate_similarity
    dpo_trainer = DPOTrainer(
        base_model,
        args=training_args,
        beta=beta, # beta
        gamma=gamma, # gamma_0
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
        # rec_model = sasrec_model,
        alpha1=alpha1, # alpha1 调节M
        alpha2=alpha2, # alpha2 调节A
        move_ratio=move_ratio, # m
        batch_shared=batch_shared,
        similarity_score = similarity_score,
        neg_accept_ratio=neg_accept_ratio,
        world_size=int(os.environ['WORLD_SIZE'])
    )
    dpo_trainer.train()
    file_name = f"{dataset}_{sample_neg_type}_dyg_gamma_{gamma}_alpha1_{alpha1}_neg_accept_ratio_{neg_accept_ratio}_batch_shared_{batch_shared}_neg_num_{neg_num}_final_checkpoint"
    output_dir = os.path.join(training_args.output_dir, file_name)
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
   
if __name__ == "__main__":
    fire.Fire(train)