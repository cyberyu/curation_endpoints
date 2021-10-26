
import utils


def get_spacy_preds(texts, model):
    """ for now texts is list of strings
    Output: list of lists of character spans along with label for each text predictions
    """
    if not type(texts) is list:
        texts = [texts]
    
    preds_list = []
    for doc in model.pipe(texts):
        preds = doc.ents

        # it's possible that this model uses different tokenization than in 'doc'
        # so need convert it predictions to character spans then back to token spans in
        # original doc
        spans = []
        for pred in preds:
            start_ind, end_ind = pred[0].idx, pred[-1].idx + len(pred[-1])
            span = doc.char_span(start_ind, end_ind, label=pred.label_)
            
            # map predicted label to Conll for demo
            lab = utils.CONLL_MAPPINGS.get(pred.label_, None)
            
            if span is not None and lab is not None:
                spans.append({'name': lab,
                              'pos': [start_ind, end_ind - 1], 
                              'tpos': [span.start,span.end - 1],
                              'text': span.text,
                              'confidence': 1
                             })
                                
        preds_list.append(spans)
    
    return preds_list

 
