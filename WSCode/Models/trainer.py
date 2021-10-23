
import torch.nn as nn
import spacy
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.utils.clip_grad import clip_grad_value_
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import time
from WSCode.Models.models import *
from WSCode.dataset import NERDataset, NERDatasetWithSpans
from WSCode.Models.dws import DWS, DWSUseSpans
from seqeval.scheme import IOB2
from seqeval.metrics import f1_score
import os
import pickle
from torch.utils.data import DataLoader

DEBUG = False
if DEBUG:
    print('DEBUGGING')

    
class Trainer:

    def __init__(self, flags):
        
        self.unpositioned_labs = ['O'] + sorted(set('-'.join(x.split('-')[1:]) for x in flags.label_list if x != 'O'))
        flags.unpositioned_labs = self.unpositioned_labs 
        self.train_iterations = flags.train_iterations
        self.batch_size = flags.batch_size
        self.flags = flags
        self.nlp = spacy.blank('en')
        self.init_model(flags)
        self.use_span_generator = (self.flags.pred_spans_model_ext is not None)

    def init_model(self,flags):
        if flags.weak_supervision_model == 'fuzzy_crf':
            Model = FuzzyLSTMCRF
        elif flags.weak_supervision_model == 'simple_baseline':
            Model = SimpleBaseline
        elif flags.weak_supervision_model == 'dws_use_spans':
            Model = DWSUseSpans
        else: 
            assert(flags.weak_supervision_model == 'dws')
            Model = DWS
        
        self.label_model = Model(flags)
        self.label_model.to(DEVICE)
        
        if self.flags.train_models:
            self.optimizer = torch.optim.RMSprop(self.label_model.parameters(), lr=flags.initial_lr)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=40, eta_min=0.00001)
            

    def get_weak_label_matrix_from_spans(self, doc, doc_weak_labels, span_model_preds):
        """Similar to get_weak_label_matrix except here we create it just for some given spans.
        For each span, we look at each labeler and check if they make a prediction on it or not.
        This method also returns labels to warmup (model preds from candidate span generator)

        """
        
        # first will get all spans from span model and map to index of span as well as token indices
        # Note: we were with character position since can't always trust token position (could be from different tokenizer)
        spans_dict = {}
        span_ind = 0
        spans_to_use = []
        warmup_labs = []
        for span in span_model_preds:
            ent_span = doc.char_span(span['pos'][0], span['pos'][1]+1)
            if ent_span is not None:
                assert((span['pos'][0], span['pos'][1]) not in spans_dict)
                spans_dict[(span['pos'][0], span['pos'][1])] = span_ind
                spans_to_use.append((ent_span.start, ent_span.end))
                warmup_labs.append(self.unpositioned_labs.index(span['name'])-1) # to align with rest of code
                span_ind += 1

        weak_lab_obs = np.zeros((len(spans_dict.keys()), len(self.flags.source_names)), dtype=np.float32)
        
        for source, preds in doc_weak_labels.items():
            if source not in self.flags.source_to_ind:
                continue
            source_ind = self.flags.source_to_ind[source]
            for pred in preds:
                span_ind = spans_dict.get((pred['pos'][0], pred['pos'][1]), None)
                if span_ind is None or pred['name'] not in self.unpositioned_labs or pred['name'] == 'O': # for now we don't allow preds of 'O'
                    continue
            
                weak_lab_obs[span_ind, source_ind] = self.unpositioned_labs.index(pred['name'])
                
        # want abstains to be -1 to be compatible with rest of code
        weak_lab_obs -= 1
        
        # now want to remove any spans with no weak labels
        has_weak_labs = (weak_lab_obs != -1).sum(axis=1) > 0
        inds_to_keep = np.argwhere(has_weak_labs).flatten()
        weak_lab_obs = weak_lab_obs[has_weak_labs]
        warmup_labs = [warmup_labs[i] for i in inds_to_keep]
        spans_to_use = [spans_to_use[i] for i in inds_to_keep]

        return weak_lab_obs, warmup_labs, spans_to_use


    def get_weak_label_matrix(self, doc, doc_weak_labels):
        """
        doc: SpaCy doc that is only used for it's tokenization
        doc_weak_labels: dictionary of labeler to list of preds
        
        For now break up return value into a seperate element per sentence
        """
        
        weak_lab_obs = np.zeros((len(doc), len(self.flags.source_names), len(self.flags.label_list)), dtype=np.float32)
        for source, preds in doc_weak_labels.items():
            if source not in self.flags.source_to_ind:
                continue
            source_ind = self.flags.source_to_ind[source]
            weak_lab_obs[:,source_ind,0] = 1.0
            for pred in preds:
                #if pred['tpos'][0] == -1: # CANT ALWAYS TRUST TOKEN POS
                #    # need to fill token positions
                pred_span = doc.char_span(pred['pos'][0], pred['pos'][1]+1) # assuming preds given include last index
                if pred_span is None:
                    continue
                
                #pred['tpos'] = [pred_span.start, pred_span.end]   # TODO FIX THIS IF MOVE AWAY FROM USING DOC

                start, end = pred_span.start, pred_span.end #pred['tpos']
                if pred['name'] not in self.unpositioned_labs or pred['name'] == 'O': # for now we don't allow preds of 'O'
                    continue

                weak_lab_obs[start:end, source_ind, 0] = 0.0
                weak_lab_obs[start, source_ind, self.flags.label_list.index('B-'+pred['name'])] = 1.0 #conf
                if end-start > 1:
                    weak_lab_obs[start+1:end, source_ind, self.flags.label_list.index('I-'+pred['name'])] = 1.0 #conf 

        #now convert into shape (doc_len, sources) where each source just takes argmax
        weak_lab_obs = np.argmax(weak_lab_obs, axis=2) 

        # want abstains to be -1 to be compatible with rest of code
        weak_lab_obs -= 1
        
        return weak_lab_obs
    
    

    def get_data(self, docs, weak_labels, span_model_preds):
        """Get weak labeler votes and bert embeddings for each token for each doc in the list of documents 
        so that we can save them and efficiently reuse them during training. 

        Arguments:
            docs {List[Spacy Doc]} 
            weak_labels  -- predictions of all the weak labelers on each doc. 

        Keyword Arguments:
            span_model_preds {List[List[Dict]]} -- predictions of the model that generates the entity spans for DWS to train on. 
                                                   When not needed, should be [[None]*len(weak_labels)]

        """
        
        all_obs = []
        all_bert_embs = []
        text = []
        all_warmup_labs = [] # predictions by span generator to use for warmup if using DWS
        all_spans_to_use = [] # list of token index spans so that we can map predictions back to actual tokens later.
        doc_offset = 0
        for doc, doc_weak_labels, span_model_pred in zip(docs, weak_labels, span_model_preds):
            warmup_labs = [None]
            spans_to_use = [None]
            if self.use_span_generator:
                weak_label_matrix, warmup_labs, spans_to_use = self.get_weak_label_matrix_from_spans(doc, doc_weak_labels, span_model_pred)
            else:
                found_one=False
                for v in doc_weak_labels.values():
                    if len(v) > 0:
                        found_one = True
                        break
                #if found_one:
                #    print('Got one')
                weak_label_matrix = self.get_weak_label_matrix(doc, doc_weak_labels)

            if len(spans_to_use) > 0:
                # is always true if not using candidate entity spans
                bert_embs = self.label_model.get_bert_embs(doc, spans_to_use)
                all_obs.append(weak_label_matrix)
                all_bert_embs.append(bert_embs)
                if len(all_obs) % 100 == 0:
                    print('Num docs processed for creating reusable embeddings: {}'.format(len(all_obs)))
                
                # update all indices in spans_to_use by doc_offset so that they are offset from first doc
                if self.use_span_generator:
                    spans_to_use = [(x[0]+doc_offset, x[1]+doc_offset) for x in spans_to_use]
                
                all_spans_to_use.extend(spans_to_use)
                all_warmup_labs.extend(warmup_labs)

            text.extend([x.text for x in doc]) # fine if this list is longer than all_obs and others
            doc_offset += len(doc)
            
        doc_lens = [len(doc) for doc in docs]
        if len(all_obs) == 0:
            all_obs = np.zeros((0,1))
            all_bert_embs = torch.zeros(0,1)
        else:
            all_obs = np.vstack(all_obs)
            all_obs = torch.from_numpy(all_obs)
            all_bert_embs = torch.cat(all_bert_embs, 0)

        return {'obs': all_obs, 
                'bert_embs': all_bert_embs, 
                'doc_lens': doc_lens, 
                'text': text, 
                'spans_to_use': all_spans_to_use, 
                'warmup_labs': all_warmup_labs}


    def label(self, weak_label_matrix, doc_bert_embs, text, spans_to_use=None):
        
        #create validation dataset 
        if self.use_span_generator:
            val_dataset = NERDatasetWithSpans(embs=doc_bert_embs, obs=weak_label_matrix)
            dataset = DataLoader(val_dataset, batch_size=64, shuffle=False)
            dataset = [x for x in dataset] # right now this is to keep compatability with old code. Fix in future.
        else: 
            chunk_sz = 500 
            dataset = NERDataset(embs=doc_bert_embs, obs=weak_label_matrix, doc_lens=[doc_bert_embs.shape[0]], 
                                chunk_size=chunk_sz, batch_num_toks=1024,
                                training_data=False, min_votes_for_token=1,
                                text=text) 

        preds_list = []
        for i in range(len(dataset)):
            obs_,embs_, text_, _ = dataset[i]
            if obs_ is None:
                break
            
            if self.use_span_generator:
                preds = self.label_model.get_preds(embs_.to(DEVICE))
                # now alter preds on examples where no weak labels to -1 (will be shifted to 0 later)
                num_weak_labs = (obs_ != -1).sum(dim=1)
                preds[num_weak_labs == 0] = -1
                preds_list.extend(preds.to('cpu').tolist())
            else:
                with torch.no_grad():
                    lstm_embs = self.label_model.run_lstm(embs_)
                    pad_mask = (lstm_embs==0).sum(dim=2) != lstm_embs.shape[2]
                    preds = self.label_model.get_crf_preds(spans=None, doc=None, embs=lstm_embs, pad_mask=pad_mask, obs=obs_)
                
                first_pad_ind = np.argmin(np.array(pad_mask.cpu()),axis=0)
                for batch_ind in range(pad_mask.shape[1]):
                    if first_pad_ind[batch_ind] == 0:
                        to_pred = preds[batch_ind]
                    else:
                        to_pred = preds[batch_ind][:first_pad_ind[batch_ind]]

                    inds_to_use = ((obs_[batch_ind]!=-1).sum(dim=1))>= 1 
                    assert(inds_to_use.shape[0] == to_pred.shape[0])   
                    to_pred[~inds_to_use] = len(self.flags.label_list)-1
                    preds_list.append(to_pred)
        
        
        maj_labels = None
        if self.flags.train_models:
            # GETTING majority vote preds as baseline to compare to **
            c = weak_label_matrix.unsqueeze(2).repeat(1, 1, self.flags.num_classes)
            eq = (c == torch.arange(self.flags.num_classes).view(1,1,-1))

            #now want to find the number of preds of each class for each example
            none_ind = self.flags.num_classes 
            max_count,argmax = eq.sum(dim=1).max(dim=1)
            maj_y_prime = torch.ones_like(max_count) * none_ind
            mask = max_count>0
            maj_y_prime[mask] = argmax[mask]
            #y_prime = y_prime.unsqueeze(2)
            maj_preds_list = maj_y_prime.tolist()
            maj_labels = [self.flags.label_list[pred_+1 if pred_!=(len(self.flags.label_list) -1) else 0] for pred_ in maj_preds_list]
        

        labels = []
        for p_ind, p in enumerate(preds_list):
            if self.use_span_generator:
                if p == -1:
                    continue
                ent_span = spans_to_use[p_ind]
                labels.extend(['O']*(ent_span[1]-len(labels)))
                # now use BI positioned labels
                unpos_lab = self.unpositioned_labs[p+1] # CHECK THIS IS RIGHT if p!=(len(self.flags.label_list) -1) else 0]
                labels[ent_span[0]] = 'B-' + unpos_lab
                for e_ind in range(ent_span[0]+1, ent_span[1]):
                    labels[e_ind] = 'I-' + unpos_lab
                
            else:
                new_preds = [self.flags.label_list[pred_+1 if pred_!=(len(self.flags.label_list) -1) else 0] for pred_ in p]
                new_preds = [(x if 'NoEnt' not in x else 'O') for x in new_preds]
                labels.extend(new_preds)
        
        confidences = np.ones(len(labels))    
        #for x in labels:
        #    if x != 'O':
        #        print('Got one in trainer...')
        return labels, confidences, maj_labels


    def train(self, val_labels, best_val_score=0):
        torch.cuda.empty_cache()
        data_path = self.flags.local_data_path 
        
        # Load in precomputed weak label observation matrices and Bert embeddings for each token
        cache_dir = os.path.join(data_path, 'cached_data')
        train_data, val_data = {}, {}
        for split_name, split_data in [['train', train_data], ['val', val_data]]:
            for d_key in ['text','doc_lens', 'spans_to_use', 'warmup_labs']:
                split_data[d_key] = json.load(open(os.path.join(cache_dir, split_name+'_'+d_key+'.json'), 'r'))
            
            split_data['bert_embs'] = torch.load(os.path.join(cache_dir, split_name + '_bert_embs.pt'))
            split_data['obs'] = torch.load(os.path.join(cache_dir, split_name+'_obs.pt'))

        print('Train obs shape: {}, train bert embs shape: {}, val obs shape: {}, val bert embs shpae: {}'.format(train_data['obs'].shape, train_data['bert_embs'].shape, val_data['obs'].shape, val_data['bert_embs'].shape))

        #create training dataset 
        if self.use_span_generator:
            train_dataset_ = NERDatasetWithSpans(embs=train_data['bert_embs'], obs=train_data['obs'], warmup_labs=train_data['warmup_labs'])
            train_dataset = DataLoader(train_dataset_, batch_size=self.flags.batch_num_spans, shuffle=True)
            train_dataset = [x for x in train_dataset]
        else:       
            train_dataset = NERDataset(embs=train_data['bert_embs'], obs=train_data['obs'], doc_lens=train_data['doc_lens'],
                                    chunk_size=self.flags.lstm_chunk_sz, batch_num_toks=self.flags.batch_num_toks,
                                    training_data=True,
                                    min_votes_for_token=self.flags.min_votes_during_warm_start,
                                    text=train_data['text'])

        # Run model for several random restarts
        for restart_num in range(self.flags.num_restarts):
            em_warmup = True
            self.init_model(self.flags)
            print('\n Starting restart: {}, best validation score: {} \n'.format(restart_num, best_val_score))
            print('Iteration to begin EM: ', self.flags.iteration_to_begin_em)
            best_iter = 0
            for train_iter in range(self.train_iterations):
                
                self.label_model.train() 
                if train_iter >= self.flags.iteration_to_begin_em:
                    em_warmup = False
                    if not self.use_span_generator:
                        train_dataset.min_votes_for_token = self.flags.min_votes_for_token
                
                stime = time.time()
                self.train_iter(train_dataset, em_warmup=em_warmup)
                
                # get validation preds
                self.label_model.eval()
                pred_labs, _, majority_vote_preds = self.label(val_data['obs'], val_data['bert_embs'], text=val_data['text'], spans_to_use=val_data['spans_to_use'])
                
                # may need to add on some 'O's to end of pred_labs if using candidate spans
                if self.use_span_generator and len(val_labels) != len(pred_labs):
                    pred_labs.extend(['O']*(len(val_labels)-len(pred_labs)))
                val_score = f1_score([val_labels], [pred_labs], scheme=IOB2, mode='strict', average='weighted')
                
                print('VAL score: ', val_score)
                print('Maj vote val score: ', f1_score([val_labels], [majority_vote_preds], scheme=IOB2, mode='strict', average='weighted'))
                #if (train_iter - best_iter > 10) or (val_score < 0.01 and self.flags.weak_supervision_model != 'fuzzy_crf'):
                #    break #go to next random restart to speed things up since likely diverged

                print('Iteration: {0}, Val_score: {1:5.4f}, LR: {2:5.4f}, Train iteration time: {3:5.4f}'.format(train_iter, val_score,
                                                                                    self.optimizer.param_groups[0]['lr'], time.time()-stime)) 
                self.scheduler.step() #Update LR

                if train_iter == self.flags.iteration_to_begin_em and self.flags.weak_supervision_model == 'dws':
                    self.optimizer = torch.optim.RMSprop(self.label_model.parameters(), lr=self.flags.initial_lr_em)
                    
                if val_score > best_val_score: 
                    best_val_score = val_score
                    best_iter = train_iter
                    print('Set best validation F1: ', best_val_score)
                    #if train_iter >= 0:
                    self.save(os.path.join(self.flags.local_data_path, self.flags.model_extension))

    
    def train_iter(self, dataset, em_warmup=False):

        for i in range(len(dataset)):

            obs, embs, text_, warmup_labs = dataset[i] #index to dataset is unneeded
            if obs is None or (self.use_span_generator and embs.shape[0] != self.flags.batch_num_spans):
                break
            #assert(len(text_) == len(obs))
            self.optimizer.zero_grad()            
            if self.use_span_generator:
                loss = self.label_model.get_loss(obs, embs, em_warmup=em_warmup, warmup_labs=warmup_labs)  
            else:
                obs = pad_sequence(obs, batch_first=False)  # (padded_seq_len, bsz, num_labelers)
                pad_mask = ((obs==0).sum(dim=2) != obs.shape[2]).to(DEVICE) #is 1 where ever no padding
                min_votes_for_token = 1  
                tokens_to_use = ((obs!=-1).sum(axis=2))>= min_votes_for_token
                tokens_to_use = tokens_to_use.to(DEVICE)
                tokens_to_use = (tokens_to_use & pad_mask)
                loss = self.label_model.get_loss(obs, embs, em_warmup=em_warmup, 
                                                pad_mask=pad_mask, tokens_to_use=tokens_to_use)
                
            loss.backward()
            #clip_grad_value_(self.label_model.parameters(), self.flags.grad_clip_value)
            self.optimizer.step()
            

    def load(self, dir_path):
        #load best label_model
        if self.flags.weak_supervision_model == 'fuzzy_crf':
            Model = FuzzyLSTMCRF
        elif self.flags.weak_supervision_model == 'dws_use_spans':
            Model = DWSUseSpans
        else:
            Model = DWS
        
        self.label_model = Model(self.flags, load_saved_model=True)
        self.label_model.load_state_dict(torch.load(os.path.join(dir_path, 'model.pt')))
        self.label_model.get_pretrained_bert()
        self.label_model.to(DEVICE)

    def save(self, dir_path):
        #save best model
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.label_model.save_model(dir_path)
        #self.label_model.get_pretrained_bert()

        # need to save flags which contains some additional info we'll need to reload the model
        with open(os.path.join(dir_path, 'flags.pkl'), 'wb') as f:
            pickle.dump(self.flags, f, protocol=pickle.HIGHEST_PROTOCOL)


    def create_reusable_embeddings(self, train_docs, train_weak_labels, val_docs, 
                                    val_weak_labels, train_span_preds, 
                                    val_span_preds):
        """ Create data that can be reused to speed up training such as BERT embeddings and weak label observation
            matrices. We don't finetune BERT so these will stay constant throughout training.

        Arguments:
            train_docs {List[SpaCy Doc]} 
            train_weak_labels  -- list of dictionaries containing weak labeler predictions for training set
            val_docs  -- same as train_docs but for validation
            val_weak_labels  -- same as train_weak_labels but for validation

        Keyword Arguments:
            train_span_preds {List[List[Dict]]} -- labels to warmup model on. This will only be used for DWS and is optional. 
                                                     If not included, then warmup will be on majority votes. 
                        
            val_span_preds {List[List[Tuple]]} -- if training DWS on spans of other model, these are the token spans
                                             to make predictions on for the validation set. This can be made for the 
                                             train set directly from train_span_preds.
            
            should set to [[None]*len(train_docs)] if not used
        """
        
        # compute validation data for future use
        val_data = self.get_data(val_docs, val_weak_labels, val_span_preds)
        
        # train_docs will be None at inference time
        if train_docs is not None:
            # only want to save if training 
            data_path = self.flags.local_data_path
            cache_dir = os.path.join(data_path, 'cached_data')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            train_data = self.get_data(train_docs, train_weak_labels, train_span_preds)

            # save these files so don't have to recompute
            for split_name, split_data in [['train', train_data], ['val', val_data]]:
                for d_key in ['text','doc_lens', 'spans_to_use', 'warmup_labs']:
                    json.dump(split_data[d_key], open(os.path.join(cache_dir, split_name+'_'+d_key+'.json'), 'w'))
                
                torch.save(split_data['bert_embs'], os.path.join(cache_dir, split_name + '_bert_embs.pt'))
                torch.save(split_data['obs'], os.path.join(cache_dir, split_name+'_obs.pt'))

            print('Saved data')

        return val_data['bert_embs'], val_data['obs'], val_data['text'], val_data['spans_to_use']



    
