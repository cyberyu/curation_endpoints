
# Can load up models here
import snips_nlu_parsers
import re 

parser = snips_nlu_parsers.BuiltinEntityParser.build(language="en")


def get_preds(texts):
    """ for now texts is list of strings
    Output: list of lists of character spans along with label for each text predictions
    """
    if type(texts) is not list:
        texts = [texts]
        
    preds_list = []
    for text in texts:
        spans = []
        # The current version of Snips has a bug that makes it crash with some rare
        # Turkish characters, or mentions of "billion years"
        #text = text.replace("’", "'").replace("”", "\"").replace("“", "\"").replace("—", "-")
        #text = text.encode("iso-8859-15", "ignore").decode("iso-8859-15")
        #text = re.sub("(\\d+) ([bm]illion(?: (?:\\d+|one|two|three|four|five|six|seven" +
        #              "|eight|nine|ten))? years?)", "\\g<1>.0 \\g<2>", text)
        try:
            results = parser.parse(text)
        except:
            results = []
            
            
        for result in results:
            text_span = text[result["range"]["start"]:result["range"]["end"]]
            #span = doc.char_span(result["range"]["start"], result["range"]["end"])
            if  text_span.lower() in {"now", "may"}:
                continue
            label = None
            if (result["entity_kind"] == "snips/number" and text_span.lower() not in
                    {"one", "some", "few", "many", "several"}):
                label = "CARDINAL"
            elif (result["entity_kind"] == "snips/ordinal" and text_span.lower() not in
                  {"first", "second", "the first", "the second"}):
                label = "ORDINAL"
            elif result["entity_kind"] == "snips/temperature":
                label = "QUANTITY"
            elif result["entity_kind"] == "snips/amountOfMoney":
                label = "MONEY"
            elif result["entity_kind"] == "snips/percentage":
                label = "PERCENT"
            elif result["entity_kind"] in {"snips/date", "snips/datePeriod", "snips/datetime"}:
                label = "DATE"
            elif result["entity_kind"] in {"snips/time", "snips/timePeriod"}:
                label = "TIME"


            if label is not None:
                spans.append({'name': label,
                              'pos': [result["range"]["start"], result["range"]["end"] - 1], 
                              'tpos': [-1,-1],
                              'text': text_span,
                              'confidence': 1
                             })
                              
        preds_list.append(spans)
    
    return preds_list
    
    
    
if __name__=='__main__':
    texts = ['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']
    preds = get_preds(texts)
    print(preds)
