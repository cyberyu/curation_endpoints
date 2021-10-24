
# Can load up models here
import spacy
from spacy_get_preds import get_spacy_preds

model = spacy.load('en_core_web_md')



def get_preds(texts):
    """ for now texts is list of strings
    Output: list of lists of character spans along with label for each text predictions
    """
    
    return get_spacy_preds(texts, model)
    
    
if __name__=='__main__':
    texts = ['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday.']
    preds = get_preds(texts)
    print(preds)
