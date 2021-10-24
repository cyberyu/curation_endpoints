
from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
import utils

# load the NER tagger
flair_model = SequenceTagger.load("flair/ner-english-ontonotes-fast")

# load small spacy model to chunk the text into sentences
spacy_model = spacy.load('en_core_web_sm', exclude=['ner'])


def get_preds(texts):
    
    if type(texts) is not list:
        texts = [texts]
        
    preds_list = []
    for text in texts:
        doc = spacy_model(text)
        sents = [[s_.text for s_ in s] for s in doc.sents]
        sentences = [Sentence(s) for s in sents]

        # predict tags for sentences
        flair_model.predict(sentences)
        
        #print('DOc texT: ', [t.text for t in doc])
        #print(spacy_model.tokenizer)
        # iterate through sentences and print predicted labels
        start_ind = 0
        new_ents = []
        for i, sentence in enumerate(sentences):
            for entity in sentence.get_spans('ner'):
                # the indexes for these spans start from 1
                indices = [t.idx for t in entity.tokens]
                #print('Entity tag: ', entity.tag)
                #print(entity)
                lab = utils.CONLL_MAPPINGS.get(entity.tag, None) # map predicted label to Conll for demo
                if lab is not None:
                    #new_ents.append(Span(doc, start=start_ind + indices[0] - 1,
                    #                 end=start_ind + indices[-1], label=entity.tag)) 
                    start_tok = doc[start_ind + indices[0] - 1]
                    end_tok = doc[start_ind + indices[-1] - 1]
                    new_ents.append({'name': lab,
                              'pos': [start_tok.idx, end_tok.idx + len(end_tok) - 1], 
                              'tpos': [-1,-1],
                              'text': text[start_tok.idx: end_tok.idx + len(end_tok)],
                              'confidence': 1
                             })
                         
            start_ind += len(sents[i])
        
        preds_list.append(new_ents)
        
    return preds_list



if __name__=='__main__':
    texts = ['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday on 1995-01-20.']
    
    import time
    stime = time.time()
    preds = get_preds(texts)
    print(time.time()-stime)
    print(preds)
