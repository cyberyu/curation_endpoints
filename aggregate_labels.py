
"""
Here we'll implement HMM and Majority vote using majority of code from: https://github.com/NorskRegnesentral/skweak/blob/main/skweak

Suppose we're given a bunch of texts with labels, need to:
 1. Create spacy docs from these with weak labels in .spans attribute
 2. Run aggregation alg
 3. Convert docs back to form we can use (ie list of predicted labels)
"""

import spacy
from typing import Dict, List, Any
from utils import extract_preds, to_docs
from skweak import aggregation, utils
import json
from utils import to_docs
import pickle
import argparse
import os
import glob
import utils

# REMOVE THIS IN FUTURE (for demo purposes). On conll, need to get rid of conll trained model (doc level uses it as well)
IGNORE_ANNOTATORS = ['core_web', 'doc_', 'doclevel']
LABELS = ['MISC', 'PER', 'LOC', 'ORG']

def train_stage2_hmm(texts: List[str], list_weak_labels: List[List[Dict[str,Any]]]):
    # first convert to format useable by our BERT training module
    from Stage2WS.utils import convert_to_huggingface
    
    full_dict = convert_to_huggingface(texts, list_weak_labels)
    
    # now train model
    
    # next publish model api for it
    

def get_preds_hmm(texts: List[str], list_weak_labels: List[Dict[str,List[Any]]]):
    docs, unique_labs = to_docs(texts, list_weak_labels, ignore_annotators=IGNORE_ANNOTATORS)
    f_ext = '/mnt/'  if __name__ == '__main__' else ''
    with open(f_ext + 'WSModels/hmm_conll.pkl', 'rb') as f:
        hmm = pickle.load(f)
    
    print('UNQIUE labels: ', unique_labs)
    docs = list(hmm.pipe(docs))
    
    # now turn back into dicts to return with list preds per doc
    preds_list = extract_preds(docs, 'hmm')
    return preds_list
    

def train_hmm(texts: List[str], list_weak_labels: List[Dict[str,List[Any]]]):
    
    docs, unique_labs = to_docs(texts, list_weak_labels, ignore_annotators=IGNORE_ANNOTATORS, labels=LABELS)
    hmm = aggregation.HMM("hmm", list(unique_labs - set(['ENT'])))
    
    hmm.fit(docs, n_iter=15) # Just run on at most 200 texts for now during training so don't run out of memory
    
    # save model for future use
    with open('/mnt/WSModels/hmm_conll.pkl', 'wb') as f:
        pickle.dump(hmm, f, protocol=pickle.HIGHEST_PROTOCOL)
        

def get_preds_majority_vote(texts: List[str], list_weak_labels: List[Dict[str,List[Any]]]):
    # try with majority vote now
    docs, unique_labs = to_docs(texts, list_weak_labels, ignore_annotators=IGNORE_ANNOTATORS, labels=LABELS)

    maj_voter = aggregation.MajorityVoterRev("majority_voter", list(unique_labs - set(['ENT'])))
    docs = list(maj_voter.pipe(docs)) #.fit_and_aggregate(docs)
    preds_list = extract_preds(docs, 'majority_voter')
    return preds_list

        
if __name__ == '__main__': 

    '''
    with open('/mnt/Data/texts.pkl', 'rb') as f:
        texts = pickle.load(f)

    with open('/mnt/Data/labels.pkl', 'rb') as f:
        labs = pickle.load(f)
    
    train_hmm(texts, labs, ignore_annotators=['core_web_md'])
    exit()
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hmm', choices=['hmm', 'majority_vote'])
    parser.add_argument('--file_ext', default='123', help="Random file extension that was used for file upload")
    args = parser.parse_args()
    
    # list files in model data for debug
    print('FILES: ', os.listdir('/mnt/ModelData/'))
    print('file ext: ', args.file_ext)
    
    '''
    with open('/mnt/Data/texts.pkl', 'rb') as f:
        texts = pickle.load(f)

    with open('/mnt/Data/labels.pkl', 'rb') as f:
        labs = pickle.load(f)
    
    for d in labs:
        for k, preds in d.items():
            for pred in preds:
                pred['pos'][1] -= 1
                pred['tpos'][1] -= 1
    '''
    texts, weak_labels, true_labels = utils.get_conll_data('', get_test_data=True)
    
    # get predictions of hmm
    print('LOGSTART')
    texts = texts[:3]
    labs = weak_labels[:3]
    #texts = texts[:2]
    #weak_labels = weak_labels[:2]
    #train_hmm(texts,labs)
    #print('Current score: 1.0')
    #print('Current score2: 4.3')
    #print('Current score 3: 5....')
    #train_hmm(texts,labs)
    hmm_preds = get_preds_hmm(texts,labs)
    print(hmm_preds[0])
    #print(hmm_preds)
    #json.dump(hmm_preds, open('/mnt/Data/hmm_preds_list.json', 'w'))
    exit()
    
    
    with open('/mnt/ModelData/texts_' + args.file_ext +'.pkl', 'rb') as f:
        texts = pickle.load(f)

    with open('/mnt/ModelData/labels_' + args.file_ext + '.pkl', 'rb') as f:
        labs = pickle.load(f)
    
    
    
    if args.model_name == 'hmm':
        preds_list = get_preds_hmm(texts, labs)
    elif args.model_name == 'majority_vote':
        preds_list = get_preds_majority_vote(texts, labs)
    else:
        raise Exception('Invalid model given')
      
    # delete labels and texts files to keep project small (eventually can do all this saving in s3 instead)
    for f in ['texts', 'labels', 'output_preds']:
        for f_ in glob.glob("/mnt/ModelData/" + f + '*'):
            os.remove(f_) #  '/mnt/ModelData/' + f + '_' + args.file_ext + '.pkl')
    
    # now save this output    
    json.dump(preds_list, open('/mnt/ModelData/output_preds_' + args.file_ext + '.json', 'w'))
    
    
