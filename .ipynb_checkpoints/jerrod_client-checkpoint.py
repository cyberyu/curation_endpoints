import requests

majvote_url="http://127.0.0.1:5000/weaksupervision/maj_vote?texts=['Joe went to Chicago.']&weak_labels=[{'crunchbase_cased': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'crunchbase_uncased': [{'name': 'ORG', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'labeler3': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}]"

response = requests.request("GET", majvote_url, headers={}, data={})
print('MajVote RESPONSE: ', response.text)

hmm_url="http://127.0.0.1:5000/weaksupervision/hmm?texts=['Joe went to Chicago.']&weak_labels=[{'crunchbase_cased': [{'name': 'PER', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}], 'crunchbase_uncased': [{'name': 'ORG', 'pos': [0,2], 'tpos': [0,0], 'text': 'Joe', 'confidence': 1}]}]"

response = requests.request("GET", hmm_url, headers={}, data={})
print('HMM RESPONSE: ', response.text)