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


if __name__ == '__main__':
    unittest.main()
