
# test where we now download model from s3, load it into memory, delete it, then run inference

from WSCode.inference import get_model_preds, get_conll_base_flags, setup_model
import time
from S3Utils.S3Utils import S3Utils
import os
import pickle
from WSCode.Models.trainer import Trainer
import spacy
import shutil

import pickle
import requests
from collections import defaultdict
import spacy 
import time
import requests
import utils
requests.packages.urllib3.disable_warnings()


url_and_auth = {'HMM': {"url": "https://domino4.domino.opsp.c1.vanguard.com:443/models/60db86494cedfd000713088c/latest/model",
                       "auth": "Myl2j0yjm40LrW3jO2rctWuoQNnbEr87qjWLF3sfiJwZrSs8RJnLiGjSky4DTooH"},
               "MajorityVote": {"url": "https://domino4.domino.opsp.c1.vanguard.com:443/models/60db7e3c4cedfd00070e9be3/latest/model",
                               "auth": "D39a0e6zN7PYhPUdPab4qDx1AA6yCcLCjmmG7wwWExYT8p07tkORIyQJBGXoKFHp"},
               "DWS": {"url": "https://domino4.domino.opsp.c1.vanguard.com:443/models/60e716ba4cedfd0008f54923/latest/model",
                      "auth": "ITRpd2wopNo2afF2llwUzyC2DBcfNVnwg0MbNQ36XHhnsFYrsECtnLo7mKIF6VII"},
               "FuzzyCRF": {"url": "https://domino4.domino.opsp.c1.vanguard.com:443/models/60e717454cedfd0008f54932/latest/model",
                           "auth": "ogWScdAnWIJnmNVziUiQnPJhcHwBaIcoy1PvKqr592BMu2VeiGs8uVHShxxMyLYI"}}



def post_request(texts_, labs_, model_to_use):
    
    response = requests.post(url_and_auth[model_to_use]['url'],
        auth=(
            url_and_auth[model_to_use]['auth'],
            url_and_auth[model_to_use]['auth']
        ),
        json={
            "data": {"texts": texts_, "list_weak_labels": labs_}

        },
        verify=False,
        
    )
    return response


texts, weak_labels, true_labels = utils.get_conll_data('', get_test_data=True)
    
    
if __name__ == '__main__':
    
    import utils
    from seqeval.scheme import IOB2
    from seqeval.metrics import f1_score
    
    nlp = spacy.load('en_core_web_sm', exclude=['ner'])
    texts, weak_labels, true_labels = utils.get_conll_data(save_folder=None, 
                                                           get_test_data=True)
    
    gold_standard = utils.convert_to_bi(nlp, texts, true_labels)
    
    # TESTER
    #texts = texts[:1]
    #weak_labels = weak_labels[:1]
    
    # get preds for model
    model_to_use = 'DWS'
    bsz = 4
    all_preds = []
    for i in range(0, 12, bsz):
        response = post_request(texts[i:i+bsz], weak_labels[i:i+bsz], model_to_use)
        json_load = response.json()
        model_preds = json_load['result']
        all_preds.extend(model_preds)
    #print('len results: ', len(results))
    
    print(all_preds[0])
    
    
    #stime2 = time.time()
    #all_preds = get_preds(texts, weak_labels)
    #print('Overall time: ', time.time()-stime2)
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
