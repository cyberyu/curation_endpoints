from WSCode.inference import get_model_preds, get_conll_base_flags, setup_model


model, nlp = setup_model(model_extension='fuzzy_crf', running_locally=(__name__ == '__main__'))

def get_preds(texts, list_weak_labels):
    return get_model_preds(texts, list_weak_labels, model, nlp)  
    

if __name__ == '__main__':
    import utils
    from seqeval.scheme import IOB2
    from seqeval.metrics import f1_score
    
    texts, weak_labels, true_labels = utils.get_test_data()
    #texts = texts
    #weak_labels = weak_labels
    #true_labels = true_labels
    texts = texts[:1]
    weak_labels = weak_labels[:1]
    true_labels = true_labels[:1]
    
    gold_standard = utils.convert_to_bi(nlp, texts, true_labels)
    
    all_preds = get_preds(texts, weak_labels)
    converted_preds = utils.convert_to_bi(nlp, texts, all_preds)
    
    # get score
    score = f1_score([gold_standard], [converted_preds], scheme=IOB2, mode='strict', average='weighted')
    print('F1: ', score)
    
    
    #for ner_preds in all_preds:
    #    print('\nHead of preds for doc:')
    #    for p in ner_preds[:5]:
    #        print(p)
