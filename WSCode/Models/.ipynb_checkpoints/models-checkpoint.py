import torch.nn as nn
import spacy
import numpy as np
import torch.nn.functional as F
import torch
import json
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer)

import logging
import time
from WSCode.Models.lstm import LSTMEncoder
from torch.nn.utils.rnn import pad_sequence
from WSCode.Models.crf import CustomCRF, FuzzyCRF
import warnings
import random
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)



class BaseModel(nn.Module):
    '''
    Defines functionality for both DWS and the Fuzzy-LSTM-CRF (both are subclasses)
    '''

    def __init__(self, flags, load_saved_model=False):
        super(BaseModel, self).__init__()
        self.flags = flags
        self.emb_dim = flags.entity_embed_sz

        if not load_saved_model and not flags.load_saved_model: # hacky solution adding to flags as well for now
            self.get_pretrained_bert()

        embed_dim = 768 if 'large' not in flags.model_to_use else 1024


        if (hasattr(self.flags, 'pred_spans_model_ext') and self.flags.pred_spans_model_ext is not None):
            # means using candidate span generator
            input_sz = embed_dim
            num_classes = len(flags.unpositioned_labs) - 1 # in this case we don't predict 'O' for now (can change later)
        else:
            input_sz = 2*flags.lstm_hidden_sz  # LSTM is bidirectional so output is twice hidden size
            num_classes = len(flags.label_list)
            self.lstm = LSTMEncoder(embed_dim=embed_dim, dropout_in=flags.dropout, dropout_out=flags.dropout,
                                    num_layers=flags.lstm_num_layers,
                                    hidden_size=flags.lstm_hidden_sz) 
            self.crf = CustomCRF(len(flags.label_list), label_list=flags.label_list, penalty=-1*self.flags.illegal_trans_penalty) 

        self.entity_transform = nn.Sequential(nn.Linear(input_sz, flags.entity_embed_sz),
                                              nn.LayerNorm(flags.entity_embed_sz))
        
        self.class_embs = nn.Embedding(num_classes, flags.entity_embed_sz)
        self.weak_labeler_embs = nn.Embedding(len(flags.source_names), flags.entity_embed_sz)
        
    
    def get_pretrained_bert(self):
        self.pretrained_bert = AutoModel.from_pretrained(self.flags.model_to_use, output_hidden_states=True).to(DEVICE)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.flags.model_to_use, use_fast=True, add_prefix_space=('roberta' in self.flags.model_to_use))
        self.pretrained_bert.eval() 
        
        
    def run_lstm(self, embs):
        src_lengths = torch.tensor([o.shape[0] for o in embs])
        padded_embs = pad_sequence(embs, batch_first=True).to(DEVICE)
        if self.flags.use_lstm:
            x, _, _ = self.lstm(padded_embs, src_lengths)
            #x has shape (seqlen, bsz, embed_dim)
            lstm_embs = torch.split(x, 1, dim=1)
            lstm_embs = [s[:src_lengths[i],0,:] for i,s in enumerate(lstm_embs)]
            padded_lstm_embs = pad_sequence(lstm_embs, batch_first=False) # (seq_len, bsz, emb_dim)
        else:
            padded_lstm_embs = padded_embs.transpose(0,1)

        return padded_lstm_embs

    
    def get_accuracies(self, embs):
        '''
        Gets accuracies of each weak labeler at each token.
        '''
        entity_embs = self.entity_transform(embs)
        score = F.linear(entity_embs, self.weak_labeler_embs.weight)
        score = score / (self.emb_dim**.5) # need to scale down the dot product for stability
        acc = torch.sigmoid(score)
        return acc, entity_embs

    def get_discr_loss(self, embs, y_prime, pad_mask, entity_embs, num_labels):
        '''
        Get loss for discriminative portion of model P(Y|X) 
        '''
        class_scores = self.get_class_scores(embs, entity_embs=entity_embs)
        class_scores = class_scores / (self.emb_dim**.5)
        loss = -self.crf(class_scores, y_prime.squeeze(2), mask=pad_mask, reduction='sum')
        return loss 


    def get_class_scores(self, embs, entity_embs=None):
        # embs is output from lstm
        if entity_embs is None:
            entity_embs = self.entity_transform(embs)

        # similarities between entity embedding and class embedding
        class_scores = F.linear(entity_embs, self.class_embs.weight)
        return class_scores

    
    def get_crf_preds(self, spans, doc, pad_mask, obs, embs):
        # expects embs to be output of lstm
        assert(embs is not None)
        
        embs = embs.to(DEVICE)
        class_scores = self.get_class_scores(embs) / (self.emb_dim**.5)
        padded_obs = pad_sequence(obs, batch_first=False)
        abstain_mask = (~(padded_obs==-1))
        num_labels = abstain_mask.sum(dim=2)
        class_scores[num_labels.unsqueeze(2).repeat(1,1,self.flags.num_classes+1)==0] = float('-inf')
        class_scores[:,:,-1][num_labels==0] = 0 #want to always predict None on examples with no weak labels
        # REMOVE 
        #class_scores[:,:,-1][num_labels > 0] = float('-inf')
        preds = self.crf.decode(class_scores, mask=pad_mask) 
        return [torch.tensor(p) for p in preds]


    def get_bert_embs(self, doc, spans_to_use=[None]):

        doc_sents = list(doc.sents)
        sentences = []
        for sent in doc_sents:
            # if sentence is too long, then split into smaller ones. Should be very rare
            for s_ind in range(0,len(sent), 150):
                sentences.append([x.text for x in sent[s_ind:s_ind+150]])
         
        tokenized_inputs = self.bert_tokenizer(
                                    sentences,
                                    padding=True, # For now just single sentence encoded at a time
                                    truncation=True,
                                    # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                                    is_split_into_words=True,
                                    return_tensors='pt'
                                ).to(DEVICE)

        hidden_state = None
        bsz = 32
        for i in range(0, len(sentences), bsz):
            with torch.no_grad():
                model_outputs = self.pretrained_bert(input_ids=tokenized_inputs['input_ids'][i:i+bsz], 
                                      attention_mask=tokenized_inputs['attention_mask'][i:i+bsz]) #  **tokenized_inputs)

            # TODO: Speed up section with batch processing
            if hidden_state is None:
                hidden_state = model_outputs.last_hidden_state.cpu()
            else:
                hidden_state = torch.clone(torch.cat([hidden_state, model_outputs.last_hidden_state.cpu()], dim=0))
        
        #print('get_bert_embs hidden shaep size: ', hidden_state.shape)

        # now need to find indices of each token in doc and how they align with bert tokenized version
        # TODO: Speed this up
        all_word_embs = []
        for sent_ind, sent in enumerate(sentences):
            #word_embs = []
            #print('Tokens: ', tokenized_inputs.tokens(sent_ind))
            for i in range(len(sent)):
                
                word_to_toks = tokenized_inputs.word_to_tokens(sent_ind, i)
                
                # there is an issue when there are extra spaces in the spacy doc (ie a token is a space).
                # In this case, just put tensor of 0's for now 
                if word_to_toks is None:
                    print('No mapping found: ') #, sent)
                    all_word_embs.append(torch.zeros(1,768))
                    continue

                word_tok_start_ind, word_tok_end_ind = word_to_toks.start, word_to_toks.end
                #print('Word: ', sent[i])
                #print('start: {}, end: {}'.format(word_tok_start_ind, word_tok_end_ind))

                # now get average embedding for each word
                
                all_word_embs.append(torch.clone(hidden_state[sent_ind, word_tok_start_ind:word_tok_end_ind, :].mean(dim=0).unsqueeze(0)))
                #all_word_embs.append(torch.clone(hidden_state[sent_ind, word_tok_start_ind, :].unsqueeze(0)))

            #all_word_embs.append(torch.cat(word_embs, dim=0))
            #print(all_word_embs[-1].shape)

        embs = None
        if spans_to_use[0] is not None:
            # want to get average embedding in each span. This condition will be true when model 
            # uses candidate entity spans from another model.
            span_embs = []
            for span in spans_to_use:
                span_embs.append(torch.cat(all_word_embs[span[0]:span[1]], dim=0).mean(dim=0).unsqueeze(0))
            
            # TODO: Fix issue when no spans present in document **
            embs = torch.cat(span_embs, dim=0) 
        else:
            embs = torch.cat(all_word_embs, dim=0) # TODO: Remove this if move to just processing sentences
        
        return embs


    def save_model(self, dir_path):
        tmp_pretrained_bert = self.pretrained_bert
        tmp_bert_tokenizer = self.bert_tokenizer
        self.pretrained_bert = None
        self.bert_tokenizer = None
        self.to('cpu')
        torch.save(self.state_dict(), os.path.join(dir_path, 'model.pt'))
        
        self.to(DEVICE)
        self.pretrained_bert = tmp_pretrained_bert
        self.bert_tokenizer = tmp_bert_tokenizer




class FuzzyLSTMCRF(BaseModel):
    
    def __init__(self, flags, load_saved_model=False):
        super(FuzzyLSTMCRF,self).__init__(flags=flags,load_saved_model=load_saved_model)
        self.crf = FuzzyCRF(flags.num_classes+1)
        
    def get_loss(self, obs, embs, pad_mask, tokens_to_use, em_warmup=False):
        embs = self.run_lstm(embs)
        obs = obs.to(DEVICE)
        
        # Create mask which has a one for any class voted on 
        # Note: obs is -1 whereever labelers abstained and also padded with 0
        obs2 = obs.unsqueeze(2).repeat(1,1,self.flags.num_classes+1,1)
        obs_eq_class = (obs2 == torch.arange(self.flags.num_classes+1).to(DEVICE).view(1,1,-1,1))
        class_mask = torch.any(obs_eq_class,dim=3) #will have 1 on any class that was voted on for that example
        
        #want class mask to have 1 on entity 'None' if no votes on any other class
        class_mask[:,:,-1] = (class_mask.sum(dim=2) == 0)
        
        # problem is that this has many false positives when only a single voter.
        # Remedy is to not look at tokens with under self.flags.min_votes_fuzzy many votes (at inference can predict 'None' on these)
        num_votes = (obs!=-1).sum(dim=2)
        class_mask[:,:,-1] = (class_mask[:,:,-1] | (num_votes < self.flags.min_votes_fuzzy))
        class_mask = (~class_mask)*(-1e7)
        entity_embs = self.entity_transform(embs)
        class_scores = self.get_class_scores(embs, entity_embs=entity_embs)
        class_scores = class_scores / (self.emb_dim**.5)
        loss = -self.crf(class_scores, class_mask, mask=pad_mask, reduction='token_mean')
        #print('Fuzzy losss: ', loss)
        return loss



  

class SimpleBaseline(nn.Module):
    def __init__(self, flags):
        super(SimpleBaseline, self).__init__()
        self.linear = nn.Linear(768, len(flags.label_list))
        self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.flags = flags

    def get_crf_preds(self, spans, doc, embs, pad_mask, obs):
        logits = self.linear(embs)
        preds = logits.argmax(dim=2)
        preds_list = []
        for i in range(preds.shape[1]):
            preds_list.append(torch.clone(preds[:,i]))
        return preds_list

    def run_lstm(self, embs):
        # fake function basically
        embs = pad_sequence(embs, batch_first=False).to(DEVICE)
        return embs

    def get_loss(self, obs, embs, em_warmup, 
                    pad_mask, tokens_to_use):
        # get majority vote preds
        c = obs.unsqueeze(3).repeat(1, 1, 1, self.flags.num_classes).to(DEVICE)
        eq = (c == torch.arange(self.flags.num_classes).view(1,1,1,-1).to(DEVICE))

        #now want to find the number of preds of each class for each example
        none_ind = self.flags.num_classes 
        max_count,argmax = eq.sum(dim=2).max(dim=2)
        y_prime = torch.ones_like(max_count) * none_ind
        mask = max_count>0
        padded_embs = pad_sequence(embs, batch_first=False).to(DEVICE)
        logits = self.linear(padded_embs)
        y_prime[mask] = argmax[mask]
        y_prime[~pad_mask] = -1
        # MASK OUT y_prime whereever it is padding ****

        # flatten logits and y_prime
        y_prime = y_prime.flatten()
        logits = logits.flatten(start_dim=0, end_dim=1)

        loss = self.loss(logits, y_prime)
        #print('Loss: ', loss)
        return loss

    
