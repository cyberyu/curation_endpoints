import unittest
import requests

host = 'http://127.0.0.1:5000'
s = requests.session()


def call_api(url, data):
    headers = {'Content-type': 'application/json'}
    r = s.post(url, json=data, headers=headers)
    if r.ok:
        return r.json()
    else:
        return r.reason


class ClientTest(unittest.TestCase):
    def test_flair(self):
        data = {
            'texts': ['I went to Victoria this holiday.',
                      'How is weather in Chicago?']
        }
        url = host + '/pretrainNER/flair'
        result = call_api(url, data)
        print(result)

    def test_zeroshot(self):
        data = {
            'texts': ['I really liked that.', 'Why was it so boring.'],
            'labels': ['travel', 'cooking', 'dancing']
        }
        url = host + '/pretrainedclassification/zero_shot'
        result = call_api(url, data)
        print(result)

    def test_movie_sentiment(self):
        url = host + '/pretrainedclassification/movie_sentiment'
        data = {
            'texts': ['I really liked that.', 'Why was it so boring.']
        }
        result = call_api(url, data)
        print(result)

    def test_fin_sentiment(self):
        url = host + '/pretrainedclassification/fin_sentiment'
        data = {
            'texts': ['Stocks rallied and the British pound gained.']
        }
        result = call_api(url, data)
        print(result)


if __name__ == '__main__':
    unittest.main()
