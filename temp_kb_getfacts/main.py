from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from extract_facts import OpenRE_get_facts
from IPython import embed

app = Flask(__name__)
api = Api(app)


class RelationExtractor(Resource):
    def __init__(self):
        self.extractor = OpenRE_get_facts()

    def post(self):
        data = request.get_json()
        texts = data['data']['texts']

        res = self.extractor.get_facts(texts)
        results = []
        for ele in res:
            for e in ele['tri']:
                relation = {}
                relation['score'] = e['c']
                relation['header'] = e['h']
                relation['header_type'] = e['h_type']
                relation['relation'] = e['r']
                relation['tail'] = e['t']
                relation['tail_type'] = e['t_type']
                results.append(relation)

        return {'result': results}


api.add_resource(RelationExtractor, '/relation')

if __name__ == '__main__':
    app.run(debug=False, port=5001)
