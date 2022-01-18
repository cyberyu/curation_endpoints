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
        print('data is '+str(data))
        texts = data['data']

        res = self.extractor.get_facts(texts)
        print('***************** res is ' + str(res))
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
                relation['h_tpos']=e['h_tpos']
                relation['t_tpos']=e['t_tpos']
                relation['r_tpos']=e['r_tpos']
                relation['h_pos']=e['h_pos']
                relation['t_pos']=e['t_pos']
                relation['r_pos']=e['r_pos']
                results.append(relation)

        return {'result': results}


api.add_resource(RelationExtractor, '/relation')

if __name__ == '__main__':
    app.run(debug=False, port=5011)
