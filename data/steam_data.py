import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

import pandas as pd
import random

class SteamData(data.Dataset):
    def __init__(self, data_dir=r'./data/ref/steam',
                 stage=None,
                 cans_num=20,
                 device=None,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=3581
        self.device = device
        self.check_files()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        cans_name=[self.item_id2name[can] for can in candidates]
        
        seq = torch.tensor(temp['seq']).to(self.device)
        len_seq = torch.tensor(temp['len_seq']).to(self.device)
        len_seq_list = temp['len_seq']
        
        # sample = {
        #     'seq': temp['seq'],
        #     'movie_seq': temp['seq_title'],
        #     'len_seq': temp['len_seq'],
        #     'seq_str': self.sep.join(temp['seq_title']),
        #     'cans': candidates,
        #     'cans_name': cans_name,
        #     'cans_str': self.sep.join(cans_name),
        #     'len_cans': self.cans_num,
        #     'item_id': temp['next'],
        #     'item_name': temp['next_item_name'],
        #     'correct_answer': temp['next_item_name']
        # }

        sample = {
            'seq': seq,
            'movie_seq': temp['seq_title'],
            'len_seq': len_seq,
            'len_seq_list': len_seq_list,
            'cans': torch.tensor(candidates).to(self.device),
            'cans_name': cans_name,
            'cans_str': '::'.join(cans_name),
            'len_cans': self.cans_num,
            'next_id': torch.tensor(temp['next']).to(self.device),
            'next_title': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample
    

    
    def negative_sampling(self,seq_unpad,next_item):
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        random.seed(next_item)
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        random.seed(next_item)
        return candidates  

    def check_files(self):
        self.item_id2name=self.get_id2name()
        if self.stage=='train':
            filename="train_data.df"
        elif self.stage=='val':
            filename="val_data.df"
        elif self.stage=='test':
            filename="test_data.df"
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  

        
    def get_id2name(self):
        game_id2name = dict()
        item_path=op.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                game_id2name[int(ll[0])] = ll[1].strip()
        return game_id2name
    
    def session_data4frame(self, datapath, game_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        
        def seq_to_title(x): 
            game_list = [self.item_id2name[i] for i in x]
            title_str = '::'.join(game_list)
            return title_str
        
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)
        def next_item_title(x): 
            return game_id2name[x]
        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        return train_data