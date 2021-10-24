import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer)
import spacy
import torch
import numpy as np
from bert_inference import get_preds as get_bert_preds

f_prefix = '/mnt/' if __name__=='__main__' else '' # want to /mnt/ if running from here but not if hosting model api
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForTokenClassification.from_pretrained(f_prefix + 'trained_models/distilbert_hmm').to(device)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

nlp = spacy.load('en_core_web_sm', exclude=['ner']) # want to align our preds with spacy's tokens (our training was in similar format where each word we only predicted it's beginning token)


def get_preds(texts):
    return get_bert_preds(texts, tokenizer, model, nlp, device)
    

if __name__ == '__main__':
    import pickle, time
    
    with open('texts.pkl', 'rb') as f:
        texts = pickle.load(f)
    
    text = texts[1065]
    #text = 'Joe Shmoe went to Calgary and visited a British cemetery. It was located in the eastern side of the Apple store. I also want to see how Victoria fairs with Calgary rain fall.'
    
    stime = time.time()
    preds = get_preds(text)
    print('TOOK: ', time.time()-stime)
    print(preds)
    
