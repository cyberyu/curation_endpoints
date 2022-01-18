import requests

relation_extraction_url="http://127.0.0.1:5011/relation"
todo = {'data':
            {"texts": 'Playoff hockey is the hardest sport to watch. Especially, when the Vancouver Canucks are playing against the Boston Bruins.'}
        }
response = requests.post(relation_extraction_url, json=todo)
print('Extracted Facts: ', response.text)
