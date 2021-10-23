

from torch.nn.utils.rnn import pad_sequence
from WSCode.Models.models import BaseModel, DEVICE
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
import itertools
import copy


#NO_ENT_CLASSES = ['NoEnt', 'CARDINAL','DATE','MONEY','ORDINAL','PERCENT','QUANTITY', 'TIME']



class DWS(BaseModel):
    
    def __init__(self, flags, load_saved_model=False):
        super(DWS,self).__init__(flags=flags,load_saved_model=load_saved_model)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')

    def get_y_prime(self, obs, lstm_embs, pad_mask):
        # Finds Y' which maximizes P(L,Y|X)
        with torch.no_grad():
            obs2 = obs.unsqueeze(2).repeat(1,1,self.flags.num_classes+1,1)
            obs_eq_class = (obs2 == torch.arange(self.flags.num_classes+1).to(DEVICE).view(1,1,-1,1))
            obs_not_eq_class = (~obs_eq_class).float()
            
            acc, entity_embs = self.get_accuracies(lstm_embs) #Has shape: (seq_len, bsz, num_labelers)
            acc = acc.unsqueeze(2)
            
            term1 = (acc ** obs_eq_class.float()) * ((1 - acc) ** obs_not_eq_class) 
            scores = torch.log(term1)
            abstain_mask = (~(obs == -1)).unsqueeze(2)
            scores = scores*abstain_mask*pad_mask.unsqueeze(2).unsqueeze(3)
            scores = scores.sum(dim=3)

            #now mask out scores for classes not in the set of weak label class votes for that ex. (or let it be none)
            #Find unique classes voted on for each ex
            mask = torch.full_like(scores, float('-inf'))
            eq = obs_eq_class.transpose(2,3)[:,:,:,:-1]
            vote_threshold = 1 
            
            num_votes = (obs!=-1).sum(dim=2)
            mask[:,:,-1][num_votes < vote_threshold] = 0  
            
            #now want to find where there is at least one 1 in dim=2 (don't want y_prime to be 
            # a class that no labeler predicted for that example)
            mask2 = eq.any(dim=2)
            mask[:,:,:-1][mask2] = 0
            scores += mask
            discr_class_scores = self.get_class_scores(lstm_embs) 
            discr_class_scores = discr_class_scores / (self.emb_dim**.5)   
            scores += discr_class_scores # the paper shows why we can do this in section 'Algorithm for Optimization'
            y_prime = self.crf.decode(scores, mask=pad_mask, use_hard_crf=True, em_argmax=True)
            y_prime = pad_sequence([torch.tensor(x) for x in y_prime]).unsqueeze(2)
            return y_prime


    def get_majority_votes(self, obs):
        c = obs.unsqueeze(3).repeat(1, 1, 1, self.flags.num_classes)
        eq = (c == torch.arange(self.flags.num_classes).view(1,1,1,-1).to(DEVICE))

        #now want to find the number of preds of each class for each example
        none_ind = self.flags.num_classes 
        max_count,argmax = eq.sum(dim=2).max(dim=2)
        y_prime = torch.ones_like(max_count) * none_ind
        mask = max_count>0
        y_prime[mask] = argmax[mask]
        y_prime = y_prime.unsqueeze(2)
        return y_prime
    
           
    def get_loss(self, obs, bert_embs, pad_mask, tokens_to_use, em_warmup=False):
        
        embs = self.run_lstm(bert_embs)
        obs = obs.to(DEVICE)
        abstain_mask = (~(obs==-1))
        
        if em_warmup:
            # Train on majority vote for the warm up stage
            y_prime = self.get_majority_votes(obs)
        else:
            y_prime = self.get_y_prime(obs, embs, pad_mask)

        y_prime = y_prime.to(DEVICE)
        plxy_ll, entity_embs, num_labels = self.get_loss_plxy(obs, y_prime, embs, tokens_to_use, abstain_mask)
        loss_discr = self.get_discr_loss(embs, y_prime, pad_mask=pad_mask, entity_embs=None, num_labels=num_labels) 
        loss = -plxy_ll + loss_discr / (1+(num_labels*tokens_to_use + tokens_to_use).sum())      
        #return loss_discr / tokens_to_use.sum() # REMOVE this *********
        return loss


    def get_loss_plxy(self, obs, y_prime, embs, tokens_to_use, abstain_mask):
        '''
        Calculates loss for P(L|X,Y) which is negative log likelihood
        '''
        obs_eq_class = (obs == y_prime)
        obs_not_eq_class = (~obs_eq_class).float()
        obs_eq_class = obs_eq_class.float()
        
        acc, entity_embs = self.get_accuracies(embs)
        term1 = (acc ** obs_eq_class) * ((1 - acc) ** obs_not_eq_class)
        ll = torch.log(term1)  
        num_labels = abstain_mask.sum(dim=2) 
        ll = ll * abstain_mask * tokens_to_use.unsqueeze(2)
        ll = ll.sum()  / (1+(num_labels*tokens_to_use + tokens_to_use).sum())  
        return ll, entity_embs, num_labels
        

    
    

class DWSUseSpans(BaseModel):
    
    def __init__(self, flags, load_saved_model=False):
        super(DWSUseSpans,self).__init__(flags=flags,load_saved_model=load_saved_model)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
             
    def get_y_prime(self, obs, bert_embs):
        # Finds Y' which maximizes P(L,Y|X)
        with torch.no_grad():
            obs2 = obs.unsqueeze(1).repeat(1,len(self.flags.unpositioned_labs)-1,1)
            obs_eq_class = (obs2 == torch.arange(len(self.flags.unpositioned_labs)-1).to(DEVICE).view(1,-1,1))
            obs_not_eq_class = (~obs_eq_class).float()
            
            acc, entity_embs = self.get_accuracies(bert_embs) #Has shape: (bsz, num_labelers)
            acc = acc.unsqueeze(1)
            
            term1 = (acc ** obs_eq_class.float()) * ((1 - acc) ** obs_not_eq_class) 
            scores = torch.log(term1)
            abstain_mask = (~(obs == -1)).unsqueeze(1)
            scores = scores*abstain_mask
            scores = scores.sum(dim=2)

            #now mask out scores for classes not in the set of weak label class votes for that ex. (or let it be none)
            #Find unique classes voted on for each ex
            mask = torch.full_like(scores, float('-inf'))
            eq = obs_eq_class.transpose(1,2) #[:,:,:-1]
            #vote_threshold = 1 
            
            num_votes = (obs!=-1).sum(dim=1)
            assert((num_votes == 0).sum() == 0)
            #mask[:,-1][num_votes < vote_threshold] = 0  
            
            #now want to find where there is at least one 1 in dim=2 (don't want y_prime to be 
            # a class that no labeler predicted for that example)
            mask2 = eq.any(dim=1)
            mask[mask2] = 0
            scores += mask
            discr_class_scores = self.get_class_scores(bert_embs) 
            discr_class_scores = discr_class_scores / (self.emb_dim**.5)   
            
            # Now this differs from CRF version
            # take log prob
            # ISSUE: this has no 'O' so one less class than above does
            discr_log_probs = F.log_softmax(discr_class_scores, dim=1)
            scores += discr_log_probs 
            y_prime = torch.argmax(scores, dim=1)
            return y_prime

    
           
    def get_loss(self, obs, bert_embs, em_warmup=False, warmup_labs=None):
        bert_embs = bert_embs.to(DEVICE)
        obs = obs.to(DEVICE)
        abstain_mask = (~(obs==-1))
        
        if em_warmup:
            y_prime = warmup_labs  #self.get_majority_votes(obs)
        else:
            y_prime = self.get_y_prime(obs, bert_embs)

        y_prime = y_prime.to(DEVICE)
        plxy_ll, entity_embs, num_labels = self.get_loss_plxy(obs, y_prime, bert_embs, abstain_mask)
        loss_discr = self.get_discr_loss(bert_embs, y_prime, entity_embs=entity_embs, num_labels=num_labels) 
        loss = -plxy_ll + loss_discr / (1+(num_labels + 1).sum())      
        #return loss_discr / tokens_to_use.sum() # REMOVE this *********
        return loss

    def get_discr_loss(self, embs, y_prime, entity_embs, num_labels):
        '''
        Get loss for discriminative portion of model P(Y|X) 
        '''
        class_scores = self.get_class_scores(embs, entity_embs=entity_embs)
        class_scores = class_scores / (self.emb_dim**.5)
        loss = self.cross_entropy_loss(class_scores, y_prime)
        #loss = -self.crf(class_scores, y_prime.squeeze(2), mask=pad_mask, reduction='sum')
        return loss

    def get_preds(self, embs):
        # for now just using discriminator but may get better performance using get_y_prime
        with torch.no_grad():
            class_scores = self.get_class_scores(embs, entity_embs=None)
            class_scores = class_scores / (self.emb_dim**.5)
            return torch.argmax(class_scores, dim=1)


    def get_loss_plxy(self, obs, y_prime, embs, abstain_mask):
        '''
        Calculates loss for P(L|X,Y) which is negative log likelihood
        '''
        obs_eq_class = (obs == y_prime.unsqueeze(1))
        obs_not_eq_class = (~obs_eq_class).float()
        obs_eq_class = obs_eq_class.float()
        
        acc, entity_embs = self.get_accuracies(embs)
        term1 = (acc ** obs_eq_class) * ((1 - acc) ** obs_not_eq_class)
        ll = torch.log(term1)  
        num_labels = abstain_mask.sum(dim=1, keepdim=True)
        ll = ll * abstain_mask # * tokens_to_use.unsqueeze(2)
        ll = ll.sum()  / (1+(num_labels + 1).sum())  # Need to figure out where the extra 1 comes from***?
        return ll, entity_embs, num_labels
        

    
