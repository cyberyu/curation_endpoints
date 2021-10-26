
import WSCode.config as config
import os
from WSCode.Models.trainer import Trainer
import pickle
import spacy
import time

def get_conll_base_flags():
    source_names = ['date_detector', 'time_detector', 'money_detector', 'proper_detector', 'infrequent_proper_detector', 'proper2_detector', 'infrequent_proper2_detector', 'nnp_detector', 'infrequent_nnp_detector', 'compound_detector', 'infrequent_compound_detector', 'misc_detector', 'legal_detector', 'company_type_detector', 'full_name_detector', 'number_detector', 'snips', 'core_web_md', 'core_web_md_truecase', 'BTC', 'BTC_truecase', 'edited_BTC', 'edited_BTC_truecase', 'edited_core_web_md', 'edited_core_web_md_truecase', 'wiki_cased', 'wiki_uncased', 'multitoken_wiki_cased', 'multitoken_wiki_uncased', 'wiki_small_cased', 'wiki_small_uncased', 'multitoken_wiki_small_cased', 'multitoken_wiki_small_uncased', 'geo_cased', 'geo_uncased', 'multitoken_geo_cased', 'multitoken_geo_uncased', 'crunchbase_cased', 'crunchbase_uncased', 'multitoken_crunchbase_cased', 'multitoken_crunchbase_uncased', 'products_cased', 'products_uncased', 'multitoken_products_cased', 'multitoken_products_uncased', 'doclevel_voter', 'doc_history_cased', 'doc_history_uncased', 'doc_majority_cased', 'doc_majority_uncased']
    source_names = [x for x in source_names if ('core_web' not in x and 'doc_' not in x and 'doclevel' not in x)]

    source_to_ind = {k:i for i,k in enumerate(source_names)}
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    flags = config.get_args()
    flags.train_models = False
    flags.source_names = source_names
    flags.label_list = label_list
    flags.num_classes = len(label_list) - 1 # don't count 'O' as class for now
    flags.source_to_ind = source_to_ind
    flags.data_path = 'Data/WSData/' #TODO Need process specific name in case multiple requests come in ***
    
    return flags


def setup_model(model_extension, running_locally=False):
 
    data_path = 'WSModels/'
    if running_locally:
        data_path = '' + data_path

    model_dir = data_path + model_extension
    with open(os.path.join(model_dir, 'flags.pkl'), 'rb') as f:
        flags = pickle.load(f)
        print(flags)

    # need to override flags.data_path 
    flags.data_path = data_path
    flags.model_extension = model_extension
    flags.load_saved_model = True
    flags.train_models = False
    
    # load best trained model
    model = Trainer(flags)
    model.load(model_dir)
    model.label_model.eval()

    '''
    # now resave model but this time on cpu
    model.label_model.pretrained_bert = None
    model.label_model.bert_tokenizer = None
    model.label_model.to('cpu')
    torch.save(model.label_model.state_dict(), best_model_file)
    print('SAVED')
    exit()
    '''

    nlp = spacy.load('en_core_web_sm', exclude=['ner', 'pos', 'lemmatizer'])
    return model, nlp


def get_model_preds(texts, list_weak_labels, model, nlp, span_preds):
    # if not using candidate entity spans, then should have
    # span_preds = [[None]*len(list_weak_labels)]

    #if type(texts) is str:
    #    texts = [texts]
    all_doc_preds = []
    docs = list(nlp.pipe(texts))
    for doc, weak_labels, span_generator_preds in zip(docs, list_weak_labels, span_preds):
 
        val_bert_embs, val_obs, val_text, val_spans_to_use = model.create_reusable_embeddings(train_docs=None, train_weak_labels=None, 
                                            train_span_preds=None, val_docs=[doc],
                                             val_weak_labels=[weak_labels], val_span_preds=[span_generator_preds])
        
        if val_obs.shape[0] == 0:
            pred_labs = []
        else:
            pred_labs, _, _ = model.label(val_obs, val_bert_embs, text=val_text, spans_to_use=val_spans_to_use)
        
        if len(pred_labs) < len(doc):
            pred_labs.extend(['O']*(len(doc)-len(pred_labs)))
        
        # now convert this to format for Annotation Tool
        cur_start_ind = -1
        cur_label = 'O'
        ner_preds = []
        for i, tok in enumerate(doc):
            
            if tok.text != val_text[i]:
                print('ERROR in tokenization *****')
            
            pred_str = pred_labs[i]
            pred_str = '-'.join(pred_str.split('-')[1:]) # want to ditch positional portion
            if pred_str != cur_label:
                if cur_label != 'O':
                    # ended an entity span
                    ner_preds.append({'name': cur_label, 
                                    'tpos': [cur_start_ind, i],
                                    'confidence': 1})
                    cur_label = 'O'
                    cur_start_ind = -1

                if pred_str != 'O':
                    # then started new ent
                    cur_label = pred_str
                    cur_start_ind = i

        # handle case of entity at end of doc
        if cur_start_ind != -1:
            ner_preds.append({'name': cur_label, 
                            'tpos': [cur_start_ind, len(doc)],
                            'confidence': 1})

        # touch up dictionary to create format AnnotationTool expects
        for ent in ner_preds:
            span_last_tok = doc[ent['tpos'][1]-1]
            span_first_tok = doc[ent['tpos'][0]]
            ent['pos'] = [span_first_tok.idx, span_last_tok.idx + len(span_last_tok) - 1] # subtract 1 since include element at last index
            ent['text'] = doc[ent['tpos'][0]:ent['tpos'][1]].text
            ent['tpos'][1] -= 1
            
        all_doc_preds.append(ner_preds)
    
    return all_doc_preds 

