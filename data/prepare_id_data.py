import pandas as pd
from movielens_data import MovielensData
from steam_data import SteamData
#from goodreads_data import GoodreadsData
from lastfm_data import LastfmData
import json
from tqdm import tqdm
import torch
from rec_models import SASRec

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def rec_rank(his_seq_ids, neg_seq_ids):
    #his_seq_ids = [name2id[i] for i in historyList]
    his_seq_ids_tensor = torch.tensor(his_seq_ids).unsqueeze(0).cuda(0)
    #neg_seq_ids = [name2id[i] for i in negative_items]
    import torch.nn.functional as F
    padded_his_seq_ids_tensor = F.pad(his_seq_ids_tensor, (0, 10 - his_seq_ids_tensor.size(1)), 'constant', dataset.padding_item_id).cuda(0)
    #neg_seq_ids_tensor = torch.tensor(neg_seq_ids).unsqueeze(0).cuda(0)
    ## 载入 sasrec 模型并对所有的item ids 进行排序
    state = torch.tensor(len(his_seq_ids)).reshape(-1).cuda(0)
    scores = sasrec_model.forward_eval(padded_his_seq_ids_tensor, state)
    #print(scores.shape)
    #print(len(neg_seq_ids))
    seq_logits = sasrec_model.cacul_h(padded_his_seq_ids_tensor, state)
    avg_item_embedding = sasrec_model.cacu_x(his_seq_ids_tensor).mean(dim=1)
    neg_seq_scores = scores[neg_seq_ids]
    sorted_indices = torch.argsort(neg_seq_scores)
    sorted_neg_items = [id2name[neg_seq_ids[i]] for i in sorted_indices]
    scores_of_sorted_neg_items = neg_seq_scores[sorted_indices]

    return sorted_neg_items, scores_of_sorted_neg_items, avg_item_embedding.reshape(-1), seq_logits.reshape(-1)

if __name__ == "__main__":

    dataset_name = 'steam'
            # 载入推荐模型
    rec_model_path = f"../rec_model/{dataset_name}.pt" 
    sasrec_model = torch.load(rec_model_path, map_location="cuda:0") 
    sasrec_model.eval()
    for name, param in sasrec_model.named_parameters():
        param.requires_grad = False
    # 载入id2name
    splits = ["val", "test"]

    #splits = ["val", "test"]
    cans_num = 20
    #data_dir = "./ref/lastfm"
    #data_dir = f"./ref/{dataset_name}"    
    for split in splits:
        #lastfm = LastfmData(data_dir=data_dir, stage=split, cans_num=cans_num)
        cans_num = 20
        data_dir = f"./ref/{dataset_name}"
        dataset = None
        id2name = None
        if dataset_name == 'lastfm':
            dataset = LastfmData(data_dir=data_dir, stage=split, cans_num=cans_num)
        elif dataset_name == 'movielens':
            dataset = MovielensData(data_dir=data_dir, stage=split, cans_num=cans_num)
        elif dataset_name == 'steam':
            dataset = SteamData(data_dir=data_dir, stage=split, cans_num=cans_num)
        id2name = dataset.get_id2name()
        print(f"{split} data length: {len(dataset)}")
        dic_lis = []
        for i in tqdm(range(len(dataset))):
            historyList = dataset[i]["movie_seq"]
            itemList = dataset[i]["cans_name"]
            trueSelection = dataset[i]["next_title"]
            cans = dataset[i]["cans"].tolist()
            negative_items = []
            negative_items_ids = []
            for item_ids, item in zip(cans, itemList):
                if item != trueSelection:
                    negative_items.append(item)
                    negative_items_ids.append(item_ids)
       
            negative_items = [item for item in itemList if item != trueSelection]
            sorted_neg_items, scores_of_sorted_neg_items, avg_item_embedding, seq_logits  = rec_rank(dataset[i]["seq"], negative_items_ids)
            # 计算 item-embedding 的 avg
            # 记录 seq 的 vector
            dic = {
                "historyList": dataset[i]["movie_seq"],
                "itemList": dataset[i]["cans_name"],
                "trueSelection": dataset[i]["next_title"],
                "seq": dataset[i]["seq"].tolist(), # tensor 是不能直接转为json的所以要进行进一步的处理
                "cans": dataset[i]["cans"].tolist(),
                "item_id": dataset[i]["next_id"].tolist(),
                "sorted_neg_items": sorted_neg_items,
                "scores_of_sorted_neg_items": scores_of_sorted_neg_items.tolist(),
                "avg_item_embedding": avg_item_embedding.tolist(),
                "seq_logits": seq_logits.tolist(),
                "dataset_id": i,
            }
            dic_lis.append(dic)
        with open(f"{dataset_name}-sft-cans20/{dataset_name}-{split}.json", "w") as f:
            json.dump(dic_lis, f, indent=4)

