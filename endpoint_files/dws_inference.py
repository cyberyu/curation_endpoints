
from WSCode.inference import get_model_preds, get_conll_base_flags, setup_model
import time


model, nlp = setup_model(model_extension='dws', running_locally=(__name__ == '__main__'))

def get_preds(texts, list_weak_labels):
    return get_model_preds(texts, list_weak_labels, model, nlp)  
    

    
    
if __name__ == '__main__':
    
    import utils
    from seqeval.scheme import IOB2
    from seqeval.metrics import f1_score
    
    texts, weak_labels, true_labels = utils.get_conll_data(save_folder='', get_test_data=True, num_val_docs=5)
    #texts = texts
    #weak_labels = weak_labels
    #true_labels = true_labels
    
    gold_standard = utils.convert_to_bi(nlp, texts, true_labels)
    
    # TESTER
    texts = texts[:1]
    weak_labels = weak_labels[:1]

    stime2 = time.time()
    all_preds = get_preds(texts, weak_labels)
    print('Overall time: ', time.time()-stime2)
    exit()

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
