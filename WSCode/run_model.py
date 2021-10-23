"""
ex run: python /mnt/WSCode/run_model.py --num_restarts 1 --train_iterations 7 --model_extension fuzzy_crf --model_name_or_path distilbert-base-uncased --weak_supervision_model fuzzy_crf --s3_data_path annotation_tool/conll_data

Example of training using candidate spans after running fuzzyCRF:
python /mnt/WSCode/run_model.py --num_restarts 1 --train_iterations 7 --model_extension dws_use_spans --model_name_or_path distilbert-base-uncased --weak_supervision_model dws_use_spans --s3_data_path annotation_tool/conll_data --pred_spans_model_ext fuzzy_crf

"""

import pickle
from WSCode.Models import trainer
from WSCode import config
import numpy as np
import spacy
import utils
from typing import Dict, List, Any
from S3Utils.S3Utils import S3Utils
import os
import json
import glob
import shutil
from WSCode.inference import get_model_preds
import time


def run_ws_model(train_texts: List[str], train_labels: List[List[Dict[str,Any]]], 
                 val_texts: List[str], val_labels: List[List[Dict[str,Any]]], 
                 flags: Any, train_span_preds: List[List[Dict[str,Any]]] = None, 
                 val_span_preds: List[List[Dict[str,Any]]] = None):

    print('LOGSTART') # needed for logger to know when to start displaying to user
    nlp = spacy.load('en_core_web_sm', exclude=['ner']) # still need sentencizer
    train_docs = list(nlp.pipe(train_texts))
    val_docs = list(nlp.pipe(val_texts))

    if train_span_preds is None:
        train_span_preds = [[None] for i in range(len(train_labels))]
        val_span_preds = [[None] for i in range(len(val_labels))]

    # convert the true labels to IOB positioned and will be just one long list
    val_gold_standard_bi = []
    num_none = 0
    for doc, all_doc_labs in zip(val_docs, val_labels):
        doc_gold_labs = all_doc_labs['gold_standard']
        doc_labs_bi, num_none_doc = utils.spans_to_bi_one_doc(doc, doc_gold_labs)
        val_gold_standard_bi.extend(doc_labs_bi)
        num_none += num_none_doc

    print('Number of gold labels that didnt match spacy token span: ', num_none)
    model = trainer.Trainer(flags)
    if not flags.reuse_embeddings:
        model.create_reusable_embeddings(train_docs=train_docs, train_weak_labels=train_labels, 
                                         val_docs=val_docs, val_weak_labels=val_labels,
                                         train_span_preds=train_span_preds, val_span_preds=val_span_preds)
        print('Finished creating reuseable embeddings')

    model.train(val_labels=val_gold_standard_bi)

    # now get predictions on train set to upload as well
    # reload best model
    model.load(os.path.join(model.flags.local_data_path, model.flags.model_extension))
    model.label_model.eval()

    assert('gold_standard' not in model.flags.source_to_ind)
    print('Beginning to get train preds...')
    stime = time.time()
    train_preds = get_model_preds(train_texts, train_labels, model, nlp, span_preds=train_span_preds)
    val_preds = get_model_preds(val_texts, val_labels, model, nlp, span_preds=val_span_preds)
    print('Time to get train preds: ', time.time() - stime)
    return train_preds, val_preds
    

if __name__=='__main__':

    # partial setup of flags to send 
    flags = config.get_args()
    flags.local_data_path = '/mnt/Data/TmpData/'
    if not os.path.exists(flags.local_data_path):
        os.makedirs(flags.local_data_path)

    # download files from s3 
    S3_OBJ = utils.create_s3_obj(flags.s3_data_path) 
    data_dict = {}
    files_to_download = ['config.json', 'train_texts.pkl', 'val_texts.pkl', 'train_labels.pkl', 'val_labels.pkl']
    if flags.pred_spans_model_ext is not None:
        # add names of files containing candidate entity span predictions to use 
        files_to_download.extend([split + '_preds_'+ flags.pred_spans_model_ext + '.json' for split in ['train','val']])
    
    for f_name in files_to_download:
        local_f_path = os.path.join(flags.local_data_path, f_name)
        S3_OBJ.download(utils.BUCKET, 
                        flags.local_data_path, 
                        os.path.join(flags.s3_data_path, f_name))
        print('Downloaded: ', f_name)
        
        # load file
        if f_name.endswith('.json'):
            with open(local_f_path, 'r') as f:
                d = json.load(f)
            
            if f_name == 'config.json':
                # setup remaing flags 
                flags.source_names = d['source_names']
                flags.label_list = d['label_list']
                flags.source_to_ind = d['source_to_ind']
                flags.num_classes = len(flags.label_list) - 1 # don't count 'O' as class for now
            else:
                f_name_to_use = 'train_span_preds' if f_name.startswith('train') else 'val_span_preds'
                data_dict[f_name_to_use] = d

        else:
            with open(local_f_path, 'rb') as f:
                obj = pickle.load(f)
        
            data_dict[f_name.split('.')[0]] = obj

    # train model
    train_preds, val_preds = run_ws_model(flags=flags, **data_dict)

    # reconnect just in case
    s3_model_dir = os.path.join('annotation_tool/Models/', flags.model_extension)
    S3_OBJ = utils.create_s3_obj(base_folder='annotation_tool/')
    try:
        S3_OBJ.make_folder(bucket, s3_model_dir)
    except:
        print('Folder for models already exists so overwriting')

    # upload model to S3 
    print('Uploading model to S3...')
    for f_name in ['flags.pkl', 'model.pt']:
        full_path = os.path.join(flags.local_data_path, flags.model_extension, f_name)
        if os.path.isfile(full_path):
            print('Uploading following file to S3: ', full_path)
            S3_OBJ.upload(full_path, s3_model_dir, utils.BUCKET)
    
    # upload model train preds
    preds_path = os.path.join(flags.local_data_path, 'train_preds_'+flags.model_extension + '.json')
    json.dump(train_preds, open(preds_path,'w'))
    S3_OBJ.upload(preds_path, flags.s3_data_path, utils.BUCKET)
    
    # upload model validation preds
    preds_path = os.path.join(flags.local_data_path, 'val_preds_'+flags.model_extension + '.json')
    json.dump(val_preds, open(preds_path,'w'))
    S3_OBJ.upload(preds_path, flags.s3_data_path, utils.BUCKET)

    # delete local data files / models. Be very careful here
    print('Removing unneeded files...')
    folder = flags.local_data_path
    assert('scratch' in folder or 'data' in folder.lower()) 
    shutil.rmtree(folder)
    
    
