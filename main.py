from typing import Dict, List, Any
import snips_nlu_parsers
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flair.data import Sentence
from flair.models import SequenceTagger
from skweak_custom import aggregation
import utils
from utils import extract_preds, to_docs
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    BertModel,
    GPT2Model)
import spacy
import torch
from pretrainedNER.bert_inference import get_preds as get_bert_preds
from pretrainedNER.spacy_get_preds import get_spacy_preds
import pickle
from WSCode.inference import get_model_preds, get_conll_base_flags, setup_model
import ast
from torch.nn import functional as F
from IPython import embed

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()


class Model:
    def __init__(self):
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        self.flair_model = SequenceTagger.load("flair/ner-english-ontonotes-fast")
        self.spacy_model = spacy.load('en_core_web_sm', exclude=['ner'])
        self.spacy_model_all = spacy.load('en_core_web_md')
        self.spacy_model_trf = spacy.load('en_core_web_trf')
        self.snips_parser = snips_nlu_parsers.BuiltinEntityParser.build(language="en")

        f_prefix = ''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.roberta_model = AutoModelForTokenClassification.from_pretrained(f_prefix + 'trained_models/distilroberta_hmm').to(device)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', use_fast=True, add_prefix_space=True)

        self.distilbert_model = AutoModelForTokenClassification.from_pretrained(f_prefix + 'trained_models/distilbert_hmm').to(device)
        self.distilbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model.eval()

        self.fin_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.fin_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.fin_model.eval()

        self.fuzzycrf_model, self.fuzzycrf_nlp = setup_model(model_extension='fuzzy_crf', running_locally=True)
        self.dws_model, self.dws_nlp =setup_model('dws', running_locally=True)

        # bertm = BertModel.from_pretrained('bert-large-cased')
        # gpt2 = GPT2Model.from_pretrained('gpt2')
        # self.rel_model_dict = {
        #     'bert': bertm,
        #     'gpt2': gpt2,
        #     # 'finbert': AutoModel.from_pretrained('ProsusAI/finbert'),
        #     # 'roberta': RobertaModel.from_pretrained('roberta-base')
        # }
        # bertt = AutoTokenizer.from_pretrained('bert-large-cased')
        # gpt2t = AutoTokenizer.from_pretrained('gpt2')
        # self.rel_tokenizer_dict = {
        #     'bert': bertt,
        #     'gpt2': gpt2t,
        #     # 'finbert': AutoTokenizer.from_pretrained('ProsusAI/finbert'),
        #     # 'roberta': RobertaTokenizer.from_pretrained('roberta-base')
        # }

        print('model loading finished!')

model = Model()


class Pretrained_Classification_Zero_Shot(Resource):
    def __init__(self):
        self.classifier = model.zero_shot_classifier

    def post(self):
        data = request.get_json()
        sentences = data['data']['texts']
        texts = [sent['text'] for sent in sentences]
        print(texts)
        # labels = data['data']['labels']
        labels = ['travel', 'computer', 'finance', 'tool']
        # text, labels = parse_texts_and_zeroshotlabels()

        label_names_to_use = [x.replace('_', ' ') for x in labels]
        label_mapping = {k:v for k,v in zip(label_names_to_use, labels)}

#         label_names_to_use = [x.replace('_', ' ') for x in label_names]
#         label_mapping = {k:v for k,v in zip(label_names_to_use, label_names)}
        #output = classifier(sentences, label_names_to_use) #candidate_labels)
        output = self.classifier(texts, labels) #candidate_labels)
        print(output)
        # convert to list of dictionaries of label_names to
        results = []
        for el in output:
            results.append({label_mapping[lab]: el['scores'][i] for i, lab in enumerate(el['labels'])})

        for i, sent in enumerate(sentences):
            sent['label'] = max(results[i], key=results[i].get)
            sent['category'] = 'intent'  # todo: hard coded

        return {'result': [sentences]}


class Pretrained_Classification_Movie_Sentiment(Resource):
    def __init__(self):
        self.tokenizer = model.tokenizer
        self.model = model.model
        self.classes = ['negative', 'positive']

    def post(self):
        data = request.get_json()
        sentences = data['data']['texts']
        texts = [sent['text'] for sent in sentences]

        out = self.tokenizer(texts, padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            preds = self.model(input_ids=torch.tensor(out['input_ids']), attention_mask=torch.tensor(out['attention_mask']))
            preds = F.softmax(preds.logits, dim=1)

        results = []
        for p in preds:
            results.append({k: p[i].item() for i,k in enumerate(self.classes)})

        for i, sent in enumerate(sentences):
            sent['label'] = max(results[i], key=results[i].get)
            sent['category'] = 'sentiment' # todo: hard coded

        return {'result': [sentences]}


class Pretrained_Classification_Fin_Sentiment(Resource):
    def __init__(self):
        self.tokenizer = model.fin_tokenizer
        self.model = model.fin_model
        self.classes = ['positive', 'negative', 'neutral']

    def post(self):
        data = request.get_json()
        sentences = data['data']['texts']
        texts = [sent['text'] for sent in sentences]

        out = self.tokenizer(texts, padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            print(torch.tensor(out['input_ids']))
            print(torch.tensor(out['attention_mask']))
            preds = self.model(input_ids=torch.tensor(out['input_ids']), attention_mask=torch.tensor(out['attention_mask']))
            preds = F.softmax(preds.logits, dim=1)

        results = []
        for p in preds:
            results.append({k:p[i].item() for i, k in enumerate(self.classes)})

        for i, sent in enumerate(sentences):
            sent['label'] = max(results[i], key=results[i].get)
            sent['category'] = 'sentiment'  # todo: hard coded

        return {'result': [sentences]}


class WeakSupervision_HMM(Resource):
    def post(self):
        IGNORE_ANNOTATORS = ['core_web', 'doc_', 'doclevel']
        LABELS = ['MISC', 'PER', 'LOC', 'ORG']

        # data = request.get_json()
        # text = data['data']['texts']
        # weak_labels = data['weak_labels']
        text, weak_labels = parse_texts_and_labels()

        docs, unique_labs = to_docs(text, weak_labels, ignore_annotators=IGNORE_ANNOTATORS)
        with open('WSModels/hmm_conll.pkl', 'rb') as f:
            hmm = pickle.load(f)

        print('UNQIUE labels: ', unique_labs)
        docs = list(hmm.pipe(docs))

        # now turn back into dicts to return with list preds per doc
        preds_list = extract_preds(docs, 'hmm')
        return {'result': preds_list}


def parse_texts_and_zeroshotlabels():
    parser = reqparse.RequestParser()
    parser.add_argument('texts', type=str, required=True)
    parser.add_argument('labels', type=str, required=True)
    args = parser.parse_args()
    text = ast.literal_eval(args['texts'])
    labels = ast.literal_eval(args['labels'])
    return text, labels


def parse_texts_and_labels():
    data = request.get_json()
    text = data['data']['texts']
    weak_labels = data['data']['list_weak_labels']
    # parser = reqparse.RequestParser()
    # parser.add_argument('texts', type=str, required=True)
    # parser.add_argument('weak_labels', type=str, required=True)
    # args = parser.parse_args()
    # text = ast.literal_eval(args['texts'])
    # weak_labels = ast.literal_eval(args['weak_labels'])
    return text, weak_labels


class WeakSupervision_MajorityVote(Resource):
    def __init__(self):
        IGNORE_ANNOTATORS = ['core_web', 'doc_', 'doclevel']
        LABELS = ['MISC', 'PER', 'LOC', 'ORG']

    def post(self):
        IGNORE_ANNOTATORS = ['core_web', 'doc_', 'doclevel']
        LABELS = ['MISC', 'PER', 'LOC', 'ORG']
        text, weak_labels = parse_texts_and_labels()

        #if type(args['texts']) is not list:
        #    texts = args['texts'].split(",")

        # try with majority vote now
        docs, unique_labs = to_docs(text, weak_labels, ignore_annotators=IGNORE_ANNOTATORS, labels=LABELS)

        maj_voter = aggregation.MajorityVoterRev("majority_voter", list(unique_labs - set(['ENT'])))
        docs = list(maj_voter.pipe(docs)) #.fit_and_aggregate(docs)
        preds_list = extract_preds(docs, 'majority_voter')
        return {'result': preds_list}


class WeakSupervision_fuzzycrf(Resource):
    def __init__(self):
        self.model = model.fuzzycrf_model
        self.nlp = model.fuzzycrf_nlp

    def post(self):
        text, weak_labels = parse_texts_and_labels()
        span_preds = [[None]*len(weak_labels)]
        result = get_model_preds(text, weak_labels, self.model, self.nlp, span_preds=span_preds)
        return {'result': result}


class WeakSupervision_dws(Resource):
    def __init__(self):
        self.model = model.dws_model
        self.nlp = model.dws_nlp

    def post(self):
        text, weak_labels = parse_texts_and_labels()
        span_preds = [[None]*len(weak_labels)]
        result = get_model_preds(text, weak_labels, self.model, self.nlp, span_preds=span_preds)
        return {'result': result}


class PretrainFinBert_HMM(Resource):
    def __init__(self):
        self.model = model.spacy_model_all

    def post(self):
        data = request.get_json()
        texts = data['data']['texts']
        return {'result': get_spacy_preds(texts, self.model)}


class PretrainNER_en_core_web_md(Resource):
    def __init__(self):
        self.model = model.spacy_model_all

    def post(self):
        data = request.get_json()
        texts = data['data']['texts']
        return {'result': get_spacy_preds(texts, self.model)}


class PretrainNER_en_core_web_trf(Resource):
    def __init__(self):
        self.model = model.spacy_model_trf

    def post(self):
        data = request.get_json()
        texts = data['data']['texts']
        return {'result': get_spacy_preds(texts, self.model)}


class PretrainNER_roberta(Resource):
    def __init__(self):
        f_prefix = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.roberta_model
        self.tokenizer = model.roberta_tokenizer
        # want to align our preds with spacy's tokens (our training was in similar format where each word we only predicted it's beginning token)
        self.nlp = model.spacy_model

    def post(self):
        data = request.get_json()
        texts = data['data']['texts']
        res = get_bert_preds(texts, self.tokenizer, self.model, self.nlp, self.device)
        return {'result': res}


class PretrainNER_distilbert(Resource):
    def __init__(self):
        f_prefix = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.distilbert_model
        self.tokenizer = model.distilbert_tokenizer
        # want to align our preds with spacy's tokens (our training was in similar format where each word we only predicted it's beginning token)
        self.nlp = model.spacy_model

    def post(self):
        data = request.get_json()
        texts = data['data']['texts']
        res = get_bert_preds(texts, self.tokenizer, self.model, self.nlp, self.device)
        return {'result': res}


class PretrainNER_FLAIR(Resource):
    def __init__(self):
        # load the NER tagger
        self.flair_model = model.flair_model

        # load small spacy model to chunk the text into sentences
        self.spacy_model = model.spacy_model

    def post(self):
        args = request.get_json()
        texts = args['data']['texts']

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

        return {'result': preds_list}


class PretrainNER_SNIPS(Resource):
    def __init__(self):
        self.parser = model.snips_parser

    def post(self):
        """ for now texts is list of strings
        Output: list of lists of character spans along with label for each text predictions
        """
        data = request.get_json()
        texts = data['data']['texts']

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

        return {'result': preds_list}


api.add_resource(PretrainNER_FLAIR, '/pretrainNER/flair')
api.add_resource(PretrainFinBert_HMM, '/pretrainNER/finbert_hmm')
api.add_resource(PretrainNER_en_core_web_trf, '/pretrainNER/en_core_web_trf')
api.add_resource(PretrainNER_en_core_web_md, '/pretrainNER/en_core_web_md')
api.add_resource(PretrainNER_SNIPS, '/pretrainNER/snips')
api.add_resource(PretrainNER_roberta, '/pretrainNER/roberta')
api.add_resource(PretrainNER_distilbert, '/pretrainNER/distilbert')

api.add_resource(WeakSupervision_dws, '/weaksupervision/dws')
api.add_resource(WeakSupervision_fuzzycrf, '/weaksupervision/fcrf')
api.add_resource(WeakSupervision_HMM, '/weaksupervision/hmm')
api.add_resource(WeakSupervision_MajorityVote, '/weaksupervision/maj_vote')

api.add_resource(Pretrained_Classification_Fin_Sentiment, '/pretrainedclassification/fin_sentiment')
api.add_resource(Pretrained_Classification_Movie_Sentiment, '/pretrainedclassification/movie_sentiment')
api.add_resource(Pretrained_Classification_Zero_Shot, '/pretrainedclassification/zero_shot')

# api.add_resource(RelationExtractor, '/relation')

if __name__ == '__main__':
    app.run(debug=False)
