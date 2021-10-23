
# test where we now download model from s3, load it into memory, delete it, then run inference

from WSCode.inference import get_model_preds, get_conll_base_flags, setup_model
import time
from aggregate_labels import get_preds_hmm
from S3Utils.S3Utils import S3Utils
import os
import pickle
from WSCode.Models.trainer import Trainer
import spacy
import shutil



nlp = spacy.load('en_core_web_sm', exclude=['ner', 'pos', 'lemmatizer'])
  
    
if __name__ == '__main__':
    
    import utils
    from seqeval.scheme import IOB2
    from seqeval.metrics import f1_score
    
    texts, weak_labels, true_labels = utils.get_conll_data(save_folder=None, 
                                                           get_test_data=True)
    #texts = texts
    #weak_labels = weak_labels
    #true_labels = true_labels
    
    gold_standard = utils.convert_to_bi(nlp, texts, true_labels)
    
    # TESTER
    #texts = texts[:1]
    #weak_labels = weak_labels[:1]

    stime2 = time.time()
    all_preds = get_preds_hmm(texts, weak_labels)
    #all_preds = get_preds(texts, weak_labels)
    print('Overall time: ', time.time()-stime2)
    #exit()

    print(all_preds[0])
    print(texts[0])
    converted_preds = utils.convert_to_bi(nlp, texts, all_preds)
    
    # get score
    score = f1_score([gold_standard], [converted_preds], scheme=IOB2, mode='strict', average='weighted')
    print('F1: ', score)
    
    '''
    for ner_preds in all_preds:
        print('\nHead of preds for doc:')
        for p in ner_preds[:5]:
            print(p)
    '''
