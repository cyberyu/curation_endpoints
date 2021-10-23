
# test where we now download model from s3, load it into memory, delete it, then run inference

from WSCode.inference import get_model_preds, get_conll_base_flags, setup_model
import time
from S3Utils.S3Utils import S3Utils
import os
import pickle
from WSCode.Models.trainer import Trainer
import spacy
import shutil

# 1. download model from s3
s3_model_path = 'annotation_tool/Models/dws_v2' #fuzzy_crf'
bucket = 'vgi-domino-prod-us-east-1-data-science-lab'
amazon_profile = 'CAI/vgi-domino-prod-us-east-1'
S3_OBJ = S3Utils(amazon_profile, bucket, s3_model_path)
tmp_local_path = '/mnt/Data/TmpData/Model/'
# make temporary local directory for the model
if not os.path.exists(tmp_local_path):
    os.makedirs(tmp_local_path)

# download model from S3
for f_name in ['flags.pkl', 'model.pt']:
    S3_OBJ.download(bucket, 
                    tmp_local_path, 
                    os.path.join(s3_model_path, f_name))
    print('Downloaded: ', f_name)

# load model 
with open(os.path.join(tmp_local_path, 'flags.pkl'), 'rb') as f:
    flags = pickle.load(f)

# need to override flags.data_path 
flags.local_data_path = '/mnt/Data/TmpData/'
flags.model_extension = 'Model'
flags.load_saved_model = True
flags.train_models = False

model = Trainer(flags)
model.load(tmp_local_path)
model.label_model.eval()


print('Removing unneeded files...')
# 4. delete local data files / models
# need to be very careful not to delete the wrong directory by accident
shutil.rmtree(tmp_local_path)


nlp = spacy.load('en_core_web_sm', exclude=['ner', 'pos', 'lemmatizer'])

def get_preds(texts, list_weak_labels):
    return get_model_preds(texts, list_weak_labels, model, nlp)  
    

    
    
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
    all_preds = get_preds(texts, weak_labels)
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
