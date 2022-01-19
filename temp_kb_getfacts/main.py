from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from extract_facts import OpenRE_get_facts
#from IPython import embed

app = Flask(__name__)
api = Api(app)


class RelationExtractor(Resource):
    def __init__(self):
        self.extractor = OpenRE_get_facts()

    def post(self):
        data = request.get_json()
        texts = data['data']['texts']
        print(texts)

        res = self.extractor.get_facts(texts)
        print('***************** res is ' + str(res))
        results = []
        for ele in res:
            for e in ele['tri']:
                relation = {}
                relation['score'] = e['c']

                head = {}
                head['pos'] = e['h_pos']
                head['tpos'] = e['h_tpos']
                head['name'] = e['h_type']
                head['text'] = e['h']

                tail = {}
                tail['pos'] = e['h_pos']
                tail['tpos'] = e['t_tpos']
                tail['name'] = e['t_type']
                tail['text'] = e['t']

                hint = {}
                hint['pos'] = e['r_pos']
                hint['tpos'] = e['r_tpos']
                hint['text'] = e['r']

                relation['head'] = head
                relation['tail'] = tail
                relation['hint'] = hint
                relation['sent_pos'] = e['sent_pos']

                # relation['header'] = e['h']
                # relation['header_type'] = e['h_type']
                # relation['relation'] = e['r']
                # relation['tail'] = e['t']
                # relation['tail_type'] = e['t_type']
                # relation['h_tpos'] = e['h_tpos']
                # relation['t_tpos'] = e['t_tpos']
                # relation['r_tpos'] = e['r_tpos']
                # relation['h_pos'] = e['h_pos']
                # relation['t_pos'] = e['t_pos']
                # relation['r_pos'] = e['r_pos']
                results.append(relation)

        print('\n\n results are', results)
        return {'result': [results]}


api.add_resource(RelationExtractor, '/relation')

if __name__ == '__main__':
    app.run(debug=False, port=5011)
