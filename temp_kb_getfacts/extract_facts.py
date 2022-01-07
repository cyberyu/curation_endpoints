from transformers import AutoTokenizer, BertModel, GPT2Model, AutoModel, RobertaTokenizer, RobertaModel
import sys, os
from os import path
import time
# eng_adj_path = path.join(path.dirname(__file__), 'english-adjectives.txt')
level_up = path.join(path.dirname(__file__), '../')
sys.path.append(level_up)
level_up = path.join(path.dirname(__file__), '../../')
sys.path.append(level_up)
level_up = path.join(path.dirname(__file__), '../../../')
sys.path.append(level_up)
# sys.path.append('/mnt/U73R/knowledge_graph/github_kg/lm_kg_construction/')

os.environ["TOKENIZERS_PARALLELISM"] = 'false'


from multiprocessing import Pool
from process.process import parse_sentence
from mapper.mapper import Map, deduplication
import argparse
import en_core_web_md
from tqdm import tqdm
import json
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class OpenRE_get_facts():
    def __init__(self):
        # use_cuda = args.use_cuda
        self.nlp = en_core_web_md.load()

        self.model_dict = {
            'bert': BertModel.from_pretrained('bert-large-cased'),
            'gpt2': GPT2Model.from_pretrained('gpt2'),
            # 'finbert': AutoModel.from_pretrained('ProsusAI/finbert'),
            # 'roberta': RobertaModel.from_pretrained('roberta-base')
        }

        self.tokenizer_dict = {
            'bert': AutoTokenizer.from_pretrained('bert-large-cased'),
            'gpt2': AutoTokenizer.from_pretrained('gpt2'),
            # 'finbert': AutoTokenizer.from_pretrained('ProsusAI/finbert'),
            # 'roberta': RobertaTokenizer.from_pretrained('roberta-base')
        }

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def fixsize_perplexity(encodings):
        device = "cpu"
        model_id = "distilgpt2"
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        stride = 128
        max_length = model.config.n_positions

        nlls = []
        for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        return ppl


    def entailment_prob(self, params):
        premise, hypothesis, _ = params
        x = self.tokenizer.encode(premise, hypothesis, return_tensors='pt',
                             truncation_strategy='only_first')
        logits = self.nli_model(x)[0]

        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]

        return prob_label_is_true.item()

    def analyze_results(self, text, results):
        '''
        analyze the results of text
        split text based on line and use this line number to match with results
        '''

        # one relation between each entity pair per line

        triplet_dict = {}
        to_test = []  # (premise, hypothesis) tuples

        for res in results:
            l = res['line'] // 2
            triplet_dict[l] = {}
            premise = text[l]
            for tri in res['tri']:
                #             print(tri)
                hypothesis = f"({tri['h']}, {' '.join(tri['r'].split('_'))}, {tri['t']})"
                ht = f"{tri['h']}-...-{tri['t']}"

                # to_test.append((premise, hypothesis, ht))

                prob = self.entailment_prob(self, [premise, hypothesis, None])
                print('premise ' + str(premise) + '   ' + str(hypothesis) + ' ' + ' prob is ' + '   ' + str(prob))

                # *.get(ht) returns the head and tail, if
                # if triplet_dict[l].get(ht) return None, new head and tail combination

                if (not triplet_dict[l].get(ht)) or (triplet_dict[l][ht]['prob'] < prob):
                    print('accepted')
                    triplet_dict[l][ht] = {
                        'prob': prob,
                        'h': tri['h'],
                        'r': tri['r'],
                        't': tri['t'],
                        'h_type': tri['h_type'],
                        't_type': tri['t_type']
                    }
                else:
                    print('rejected')

                # print(f'text: {premise[:20]}, hyp: {hypothesis}, prob: {prob}')

        # use pool over all premise, hypothesis, head-tail pairs
        # modified_output = []
        # with Pool(2) as pool:  # sb: 10 -> 2 to use less memory
        #     params = to_test
        #     for output in pool.imap_unordered(entailment_prob, params):
        #         modified_output.append([params, output])
        #     pool.close()  # added by sb
        #     pool.join()  # added by sb

        output = []
        for l, tris in triplet_dict.items():
            tri_list = [v for k, v in tris.items()]
            output.append({'line': l, 'tri': tri_list})

        return output

        # return triplet_dict
        # return modified_output

    def get_facts(self, input_text, model_type='bert', use_filter=False, tree_weight=0.0, debug=False, filter=False):
        """
        function to make call depending on the arguments in info
        """
        self.info = dict()
        #     input_text = info['text']
        #     print('input text:', input_text)

        self.use_cuda = self.info.get('use_cuda', False)
        self.include_sentence = self.info.get('include_sentence', False)
        self.threshold = self.info.get('threshold', 0.003)
        # language_model = info.get('language_model', 'bert-large-cased')
        # set language model based on model_type
        self.model_mapping = {
            'bert': 'bert-large-cased',
            'gpt2': 'gpt2',
            # 'finbert': 'finbert',
            # 'roberta': 'roberta-base'
        }
        self.language_model = self.model_mapping[model_type]
        self.outputs = list()
        #     tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.tokenizer = self.tokenizer_dict[model_type]
        self.encoder = self.model_dict[model_type]
        #     if 'gpt2' in language_model:
        #         encoder = GPT2Model.from_pretrained(language_model)

        #     else:
        #         encoder = BertModel.from_pretrained(language_model)

        debug = True

        self.encoder.eval()
        if self.use_cuda:
            self.encoder = self.encoder.cuda()

        if not isinstance(input_text, list):
            input_text = [input_text]

        input_text = [a for t in input_text for a in t.split('\n')]

        for idx, line in enumerate(input_text):
            sentence = line.strip()
            #         print(f'line: {sentence}')
            if len(sentence):
                valid_triplets = []
                for sent in self.nlp(sentence).sents:
                    if debug: print(f'sentence: {sent}')
                    # Match
                    for triplets in parse_sentence(sent.text, self.tokenizer, self.encoder, self.nlp, use_cuda=self.use_cuda,
                                                   use_filter=use_filter, tree_weight=tree_weight, debug=debug):
                        valid_triplets.append(triplets)
                if debug: print('valid triplets:', valid_triplets)
                if len(valid_triplets) > 0:
                    # Map
                    mapped_triplets = []
                    for triplet in valid_triplets:
                        head = triplet['h']
                        tail = triplet['t']
                        relations = triplet['r']
                        conf = triplet['c']
                        head_type = triplet['h_type']
                        tail_type = triplet['t_type']
                        if debug: print(f'conf: {conf}, threshold: {self.threshold}')
                        if conf < self.threshold:
                            continue
                        if debug: print('Calling Map...')
                        mapped_triplet = Map(head, relations, tail, debug=True)
                        if 'h' in mapped_triplet:
                            mapped_triplet['c'] = conf
                            mapped_triplet['h_type'] = head_type
                            mapped_triplet['t_type'] = tail_type
                            mapped_triplets.append(mapped_triplet)
                    if debug: print('mapped triplets:', mapped_triplets)
                    output = {'line': idx, 'tri': deduplication(mapped_triplets)}
                    #                 print(f'output for line {idx}:', output)

                    if self.include_sentence:
                        output['sent'] = sentence
                    if len(output['tri']) > 0:
                        self.outputs.append(output)
        #                     print('output:',output)
        #                   g.write(json.dumps( output )+'\n')

        # process all outputs based on an entailment model
        if filter:
            refined_output = self.analyze_results(self, text=input_text, results=self.outputs)

            self.outputs = refined_output

        # convert all outputs to a string
        # if debug: print('outputs:', outputs)
        # new_outputs = list()
        # for out in outputs:
        #     new_tri = list()
        #     for fact in out['tri']:
        #         new_tri.append({'h': str(fact['h']),
        #                        'r': str(fact['r']),
        #                        't': str(fact['t']),
        #                        'conf': fact['c']})
        #     new_out = out.copy()
        #     new_out['tri'] = new_tri
        #     new_outputs.append(new_out)

        #     return input_text
        #     new_outputs = ['test_output1'] + new_outputs

        return self.outputs

if __name__ == '__main__':
    ogf = OpenRE_get_facts()
    res = ogf.get_facts('Playoff hockey is the hardest sport to watch. Especially, when the Vancouver Canucks are playing against the Boston Bruins.')
    # res contains the extract triplets of entity_relations
    print(res)

    # expand and extract them one by one
    #for ele in res.json()['result']:
    for ele in res:
        for e in ele['tri']:
            print('An extracted triplet has c_score {}, header_entity {}, header_type {}, relation {}. tail_entity {}, tail_type {}'.format(
                e['c'], e['h'], e['h_type'], e['r'], e['t'], e['t_type']))
