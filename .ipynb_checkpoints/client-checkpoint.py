import requests

#url = "http://127.0.0.1:5000/pretrainNER/flair?texts=['I went to Victoria this holiday.', 'How is weather in Chicago?']"
#url = "http://127.0.0.1:5000/pretrainNER/snips?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
#url = "http://127.0.0.1:5000/pretrainNER/distillbert?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
#url = "http://127.0.0.1:5000/pretrainNER/roberta?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
url = "http://127.0.0.1:5000/pretrainNER/en_core_web_md?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
#url = "http://127.0.0.1:5000/pretrainNER/en_core_web_trf?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"

#url="http://127.0.0.1:5000/weaksupervision/dws?texts='Joe went to Chicago.'&weak_labels={'labeler1': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'labeler2': [{'name': 'ORG', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'labeler3': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}"


majvote_url="http://127.0.0.1:5000/weaksupervision/fcrf?texts=['Joe went to Chicago.']&weak_labels=[{'crunchbase_cased': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'crunchbase_uncased': [{'name': 'ORG', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'labeler3': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}]"

response = requests.request("GET", majvote_url, headers={}, data={})
print('MajVote RESPONSE: ', response.text)


dws_url="http://127.0.0.1:5000/weaksupervision/dws?texts=['Joe went to Chicago.']&weak_labels=[{'crunchbase_cased': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'crunchbase_uncased': [{'name': 'ORG', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}]" #, 'labeler3': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}]"

response = requests.request("GET", dws_url, headers={}, data={})
print('DWS RESPONSE: ', response.text)

fcrf_url="http://127.0.0.1:5000/weaksupervision/fcrf?texts=['Joe went to Chicago.']&weak_labels=[{'crunchbase_cased': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'crunchbase_uncased': [{'name': 'ORG', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}]" #, 'labeler3': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}]"

response = requests.request("GET", fcrf_url, headers={}, data={})
print('Fuzzy CRF RESPONSE: ', response.text)




