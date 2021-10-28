
from torch.utils.data import Dataset
import json
import torch
import numpy as np


class NERDataset(Dataset):
    '''
    Creates minibatches for the models
    '''
    def __init__(self, embs, obs, doc_lens, chunk_size, batch_num_toks, 
                training_data=True, min_votes_for_token=1, text=None):
        
        self.min_votes_for_token = min_votes_for_token
        self.training = training_data
        self.batch_num_toks = batch_num_toks #approximately this many tokens per mini batch
        
        #now create chunks
        chunk_lens = []
        for l in doc_lens:
            chunk_lens.extend([chunk_size for _ in range(l // chunk_size)] + [l % chunk_size])
    
        self.chunk_lens = [x for x in chunk_lens if x != 0]
        cum_sum = np.cumsum(self.chunk_lens)
        self.split_text = []
        for i in range(len(cum_sum)):
            if i == 0:
                self.split_text.append(text[:cum_sum[0]])
            else:
                self.split_text.append(text[cum_sum[i-1]:cum_sum[i]])

        self.split_obs = torch.split(obs, self.chunk_lens, dim=0)
        self.split_embs = torch.split(embs, self.chunk_lens, dim=0)
        self.cur_index = 0
        
    def __len__(self):
        return len(self.chunk_lens) #Is an upper bound (user should stop looping when Nones are returned)

    def __getitem__(self, idx):
        if self.cur_index >= self.__len__():
            self.cur_index = 0
            return None, None, None, None
        
        if self.cur_index == 0 and self.training:
            #At start of epoch do shuffle of data while attempting
            # to minimize padding:
            # 1. sort by length
            # 2. break into 100 chunks and sort within each chunk
            rand_order = np.argsort(np.array(self.chunk_lens))
            num_splits = 100 
            step_sz = int(len(rand_order) / num_splits)
            for i in range(0,len(rand_order),step_sz):
                np.random.shuffle(rand_order[i:i+step_sz])
            
            self.split_text = [self.split_text[i] for i in rand_order]
            self.split_obs = [self.split_obs[i] for i in rand_order]
            self.split_embs = [self.split_embs[i] for i in rand_order]
            
        #want to keep adding to batch until at least self.batch_num_toks
        num_toks = 0
        obs = []
        embs = []
        texts = []
        while num_toks < self.batch_num_toks and self.cur_index < self.__len__():
            
            #don't use data where we won't make any preds unless validation time
            obs_ = self.split_obs[self.cur_index]
            num_to_vote_on = (((obs_!=-1).sum(axis=1))>= self.min_votes_for_token).sum().item()
            if num_to_vote_on == 0 and self.training:
                self.cur_index += 1
                continue
            
            texts.append(self.split_text[self.cur_index])
            obs.append(obs_)
            embs.append(self.split_embs[self.cur_index])
            num_toks += num_to_vote_on 
            self.cur_index += 1

        if self.training and num_toks < self.batch_num_toks:
            self.cur_index = 0
            return None, None, None, None
        
        return obs, embs, texts, None


class NERDatasetWithSpans(Dataset):
    '''
    Creates minibatches for the models where each example can be thought of as an independent token span. 
    NOTE: For now ignoring text
    '''
    def __init__(self, embs, obs, warmup_labs=None):
        self.split_obs = torch.split(obs, [1]*obs.shape[0], dim=0)
        self.split_embs = torch.split(embs, [1]*embs.shape[0], dim=0)
        if warmup_labs is None:
            warmup_labs = [-1] * len(self.split_embs)
        self.warmup_labs = warmup_labs
        #self.texts = [None]*len(self.split_obs) # keeping for compatibility with previous code but remove in future
        self.min_votes_for_token = None # to be consistent with other dataset.

    def __len__(self):
        return len(self.split_obs)    
    
    def __getitem__(self, idx):
        return self.split_obs[idx].squeeze(0), self.split_embs[idx].squeeze(0), -1, self.warmup_labs[idx]
        
