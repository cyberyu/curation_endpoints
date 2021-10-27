import requests

#url = "http://127.0.0.1:5000/pretrainNER/flair?texts=['I went to Victoria this holiday.', 'How is weather in Chicago?']"
#url = "http://127.0.0.1:5000/pretrainNER/snips?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
#url = "http://127.0.0.1:5000/pretrainNER/distillbert?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
#url = "http://127.0.0.1:5000/pretrainNER/roberta?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
#url = "http://127.0.0.1:5000/pretrainNER/en_core_web_md?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"
url = "http://127.0.0.1:5000/pretrainNER/en_core_web_trf?texts=['Calgary is hot in the July.', 'Fred went to play Soccer in the Olympic Oval yesterday and scored 4 goals in five minutes.']"

payload={}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
