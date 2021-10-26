from flask import Flask
from flask_restful import Resource, Api, reqparse
#from pretrainedNER.snips_endpoint import get_preds
import snips_nlu_parsers
from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
import utils
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer)
import spacy
import torch
import numpy as np
from pretrainedNER.bert_inference import get_preds as get_bert_preds
from pretrainedNER.spacy_get_preds import get_spacy_preds

app = Flask(__name__)
api = Api(app)


class PretrainNER_en_core_web_md(Resource):
    def __init__(self):
        self.model = spacy.load('en_core_web_md')

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=str, required=True)
        args = parser.parse_args()

        texts = args['texts']
        return get_spacy_preds(texts, self.model)


class PretrainNER_en_core_web_trf(Resource):
    def __init__(self):
        self.model = spacy.load('en_core_web_trf')

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=str, required=True)
        args = parser.parse_args()

        texts = args['texts']
        return get_spacy_preds(texts, self.model)


class PretrainNER_roberta(Resource):
    def __init__(self):
        f_prefix = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForTokenClassification.from_pretrained(f_prefix + 'trained_models/distilroberta_hmm').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', use_fast=True, add_prefix_space=True)

        self.nlp = spacy.load('en_core_web_sm', exclude=[
            'ner'])  # want to align our preds with spacy's tokens (our training was in similar format where each word we only predicted it's beginning token)


    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=str, required=True)
        args = parser.parse_args()

        texts = args['texts']

        return get_bert_preds(texts, self.tokenizer, self.model, self.nlp, self.device)

class PretrainNER_distillbert(Resource):
    def __init__(self):

        f_prefix = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForTokenClassification.from_pretrained(f_prefix + 'trained_models/distilbert_hmm').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

        self.nlp = spacy.load('en_core_web_sm', exclude=[
            'ner'])  # want to align our preds with spacy's tokens (our training was in similar format where each word we only predicted it's beginning token)


    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=str, required=True)
        args = parser.parse_args()

        texts = args['texts']

        # if type(args['texts']) is not list:
        #     texts = args['texts'].split(",")
        #     #texts = [texts]

        return get_bert_preds(texts, self.tokenizer, self.model, self.nlp, self.device)

class PretrainNER_FLAIR(Resource):

    def __init__(self):

        # load the NER tagger
        self.flair_model = SequenceTagger.load("flair/ner-english-ontonotes-fast")

        # load small spacy model to chunk the text into sentences
        self.spacy_model = spacy.load('en_core_web_sm', exclude=['ner'])

    def get(self):

        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=str, required=True)
        args = parser.parse_args()


        if type(args['texts']) is not list:
            texts = args['texts'].split(",")
            #texts = [texts]

        preds_list = []
        for text in texts:
            doc = self.spacy_model(text)
            sents = [[s_.text for s_ in s] for s in doc.sents]
            sentences = [Sentence(s) for s in sents]

            # predict tags for sentences
            self.flair_model.predict(sentences)

            # print('DOc texT: ', [t.text for t in doc])
            # print(spacy_model.tokenizer)
            # iterate through sentences and print predicted labels
            start_ind = 0
            new_ents = []
            for i, sentence in enumerate(sentences):
                for entity in sentence.get_spans('ner'):
                    # the indexes for these spans start from 1
                    indices = [t.idx for t in entity.tokens]
                    # print('Entity tag: ', entity.tag)
                    # print(entity)
                    lab = utils.CONLL_MAPPINGS.get(entity.tag, None)  # map predicted label to Conll for demo
                    if lab is not None:
                        # new_ents.append(Span(doc, start=start_ind + indices[0] - 1,
                        #                 end=start_ind + indices[-1], label=entity.tag))
                        start_tok = doc[start_ind + indices[0] - 1]
                        end_tok = doc[start_ind + indices[-1] - 1]
                        new_ents.append({'name': lab,
                                         'pos': [start_tok.idx, end_tok.idx + len(end_tok) - 1],
                                         'tpos': [-1, -1],
                                         'text': text[start_tok.idx: end_tok.idx + len(end_tok)],
                                         'confidence': 1
                                         })

                start_ind += len(sents[i])

            preds_list.append(new_ents)

        return preds_list


class PretrainNER_SNIPS(Resource):

    def __init__(self):

        self.parser = snips_nlu_parsers.BuiltinEntityParser.build(language="en")


    # def get(self):
    #     return {'hello': 'world'}


    def get(self):
        """ for now texts is list of strings
        Output: list of lists of character spans along with label for each text predictions
        """

        parser = reqparse.RequestParser()
        parser.add_argument('texts', type=str, required=True)
        args = parser.parse_args()


        if type(args['texts']) is not list:
            texts = args['texts'].split(",")
            #texts = [texts]

        preds_list = []
        for text in texts:
            spans = []
            # The current version of Snips has a bug that makes it crash with some rare
            # Turkish characters, or mentions of "billion years"
            # text = text.replace("’", "'").replace("”", "\"").replace("“", "\"").replace("—", "-")
            # text = text.encode("iso-8859-15", "ignore").decode("iso-8859-15")
            # text = re.sub("(\\d+) ([bm]illion(?: (?:\\d+|one|two|three|four|five|six|seven" +
            #              "|eight|nine|ten))? years?)", "\\g<1>.0 \\g<2>", text)
            try:
                results = self.parser.parse(text)
            except:
                results = []

            for result in results:
                text_span = text[result["range"]["start"]:result["range"]["end"]]
                # span = doc.char_span(result["range"]["start"], result["range"]["end"])
                if text_span.lower() in {"now", "may"}:
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
                                  'tpos': [-1, -1],
                                  'text': text_span,
                                  'confidence': 1
                                  })

            preds_list.append(spans)

        return preds_list


api.add_resource(PretrainNER_SNIPS, '/pretrainNER/snips')
api.add_resource(PretrainNER_FLAIR, '/pretrainNER/flair')
api.add_resource(PretrainNER_distillbert, '/pretrainNER/distillbert')
api.add_resource(PretrainNER_roberta, '/pretrainNER/roberta')
api.add_resource(PretrainNER_en_core_web_md, '/pretrainNER/en_core_web_md')
api.add_resource(PretrainNER_en_core_web_trf, '/pretrainNER/en_core_web_trf')


if __name__ == '__main__':
    app.run(debug=True)