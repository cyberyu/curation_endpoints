import spacy
import pickle
import time
from spacy.tokens import Doc
from typing import Dict, List, Any
import copy
import os
import json
import torch
import re
from copy import copy
from collections import defaultdict
import numpy as np

alphabet = re.compile(r'^[a-zA-Z]+$')


CONLL_TO_RETAIN = {"PER", "MISC", "ORG", "LOC"}

CONLL_MAPPINGS = {"PERSON": "PER", "COMPANY": "ORG", "GPE": "LOC", 'EVENT': "MISC", 'FAC': "MISC", 'LANGUAGE': "MISC",
                  'LAW': "MISC", 'NORP': "MISC", 'PRODUCT': "MISC", 'WORK_OF_ART': "MISC"}

nlp = spacy.blank('en')

def to_docs(texts, labs, ignore_annotators=[], labels=[]):
    unique_labs = set()
    docs = list(nlp.pipe(texts))

    # now need to add spangroups
    for doc, labs_ in zip(docs, labs):
        for annotator, span_list in labs_.items():
            if any(x in annotator for x in
                   ignore_annotators):  # ignore_annotators contains subsets of what we want to match
                continue

            doc.spans[annotator] = []
            for span in span_list:
                # if span['name'] not in labels:
                #    continue

                s = doc.char_span(span['pos'][0], span['pos'][1] + 1,
                                  label=span['name'])  # span['pos'] includes final index so +1 here for slice indices
                if s is not None:
                    doc.spans[annotator].append(s)
                    unique_labs.add(span['name'])
    # print(docs[0].spans)
    return docs, unique_labs


def from_docs(docs, ignore_annotators=[]):
    docs_text = []
    docs_labs = []
    for doc in docs:
        span_groups = doc.spans
        docs_text.append(doc.text)
        labs = {}
        for annotator, span_list in span_groups.items():
            if any(x in annotator for x in
                   ignore_annotators):  # ignore_annotators contains subsets of what we want to match
                continue

            # print(annotator)
            # print(span_list)
            labs[annotator] = [{'name': span.label_,
                                'pos': [span[0].idx, span[-1].idx + len(span[-1])],
                                'tpos': [span.start, span.end],
                                'text': span.text,
                                'confidence': 1
                                } for span in span_list]

        docs_labs.append(labs)

    return docs_text, docs_labs


def extract_preds(docs, model_name):
    all_preds = []
    for doc in docs:
        if model_name not in doc.spans:
            all_preds.append([])
            continue

        # make sure indices for 'pos' and 'tpos' are such that they are inclusive of end index
        all_preds.append([{'name': span.label_,
                           'pos': [span[0].idx, span[-1].idx + len(span[-1]) - 1],
                           'tpos': [span.start, span.end - 1],
                           'text': span.text,
                           'confidence': 1
                           } for span in doc.spans[model_name]])

    return all_preds


def get_conll_data(save_folder, num_val_docs=10, num_test_docs=100):
    # NOTE: these saved files have weak labels and true labels with spans that don't include the final index. The code for annotationTool expects
    # that predicted spans have last index of span be inclusive. Therefore we alter that below. 
    # Assumes locations of the data files are known and does the loading / preprocessing
    weak_labels_file = '/mnt/Data/labels.pkl'
    true_labels_file = '/mnt/Data/true_labels.pkl'
    texts_file = '/mnt/Data/texts.pkl'

    with open(weak_labels_file, 'rb') as f:
        weak_labels = pickle.load(f)

    with open(true_labels_file, 'rb') as f:
        true_labels = pickle.load(f)

    with open(texts_file, 'rb') as f:
        texts = pickle.load(f)

    # print(weak_labels[0].keys())
    # print(true_labels[0][:10])
    # print(texts[0][:50])

    # alter all true labels and weak labels by lowering 'pos' end indice by one 
    for d in true_labels:
        for l in d:
            l['pos'][1] -= 1
            l['tpos'][1] -= 1

    for d in weak_labels:
        for k, preds in d.items():
            for pred in preds:
                pred['pos'][1] -= 1
                pred['tpos'][1] -= 1

    '''
    if get_test_data:
        # want to ignore final 'num_val_docs' docs when getting docs to test on since used in validation. 
        orig_len = len(texts)
        inds_to_ignore = [len(texts)-i for i in range(1,num_val_docs+1)] 
        texts = [texts[i] for i in range(orig_len) if i not in inds_to_ignore]
        labels = [weak_labels[i] for i in range(orig_len) if i not in inds_to_ignore]
        true_labels = [true_labels[i] for i in range(orig_len) if i not in inds_to_ignore]
        return texts, labels, true_labels
    '''

    # shuffle data
    np.random.seed(1)  # so shuffle is same each time
    ind_order = [i for i in range(len(true_labels))]
    np.random.shuffle(ind_order)
    weak_labels = [weak_labels[i] for i in ind_order]
    true_labels = [true_labels[i] for i in ind_order]
    texts = [texts[i] for i in ind_order]

    data_dict = {}
    data_dict['val_texts'] = texts[:num_val_docs]
    data_dict['val_labels'] = weak_labels[:num_val_docs]
    data_dict['val_true_labels'] = true_labels[:num_val_docs]

    data_dict['test_texts'] = texts[num_val_docs:(num_val_docs + num_test_docs)]
    data_dict['test_true_labels'] = true_labels[num_val_docs: (num_val_docs + num_test_docs)]

    data_dict['train_texts'] = texts[(num_val_docs + num_test_docs):]
    data_dict['train_labels'] = weak_labels[(num_val_docs + num_test_docs):]
    data_dict['train_true_labels'] = true_labels[(num_val_docs + num_test_docs):]

    print('Num tokens for val: ', sum(len(x.split(' ')) for x in data_dict['val_texts']))
    print('Num tokens for test: ', sum(len(x.split(' ')) for x in data_dict['test_texts']))

    # now fix up labels by adding gold standard in 
    # REmove this eventually. Right now keeps compatibility with DWS 
    for i, d in enumerate(data_dict['val_labels']):
        d['gold_standard'] = data_dict['val_true_labels'][
            i]  # doing this for now only for compatibility with other code

    # NEED SOURCE_NAMES to use (Need to make sure always in same order)
    # source_names = set()
    # for d in weak_labels:
    #    source_names = source_names.union(set(d.keys()))
    # source_names = sorted([x for x in source_names if 'core_web' not in x])

    source_names = ['date_detector', 'time_detector', 'money_detector', 'proper_detector', 'infrequent_proper_detector',
                    'proper2_detector', 'infrequent_proper2_detector', 'nnp_detector', 'infrequent_nnp_detector',
                    'compound_detector', 'infrequent_compound_detector', 'misc_detector', 'legal_detector',
                    'company_type_detector', 'full_name_detector', 'number_detector', 'snips', 'core_web_md',
                    'core_web_md_truecase', 'BTC', 'BTC_truecase', 'edited_BTC', 'edited_BTC_truecase',
                    'edited_core_web_md', 'edited_core_web_md_truecase', 'wiki_cased', 'wiki_uncased',
                    'multitoken_wiki_cased', 'multitoken_wiki_uncased', 'wiki_small_cased', 'wiki_small_uncased',
                    'multitoken_wiki_small_cased', 'multitoken_wiki_small_uncased', 'geo_cased', 'geo_uncased',
                    'multitoken_geo_cased', 'multitoken_geo_uncased', 'crunchbase_cased', 'crunchbase_uncased',
                    'multitoken_crunchbase_cased', 'multitoken_crunchbase_uncased', 'products_cased',
                    'products_uncased', 'multitoken_products_cased', 'multitoken_products_uncased', 'doclevel_voter',
                    'doc_history_cased', 'doc_history_uncased', 'doc_majority_cased', 'doc_majority_uncased']
    source_names = [x for x in source_names if ('core_web' not in x and 'doc_' not in x and 'doclevel' not in x)]
    source_to_ind = {k: i for i, k in enumerate(source_names)}
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    config = {'source_names': source_names,
              'source_to_ind': source_to_ind,
              'label_list': label_list}

    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        json.dump(config, open(os.path.join(save_folder, 'config.json'), 'w'))

        for k, v in data_dict.items():
            with open(os.path.join(save_folder, k + '.pkl'), 'wb') as f:
                pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)

    return config, data_dict


def spans_to_bi_one_doc(doc: Doc, preds: List[Dict[str, Any]]) -> List[str]:
    doc_labs = ['O'] * len(doc)
    num_none = 0
    for pred in preds:
        # ex: {'name': 'ORG', 'pos': [0, 1], 'tpos': [0, 0], 'text': 'EU', 'confidence': 1}
        lab_span = doc.char_span(pred['pos'][0], pred['pos'][1] + 1)  # last index is included in pred
        if lab_span is None:
            num_none += 1
            if DEBUG:
                print('Pred resulted in span of none: ', doc.text[pred['pos'][0]:pred['pos'][1] + 1])
                print('region around span: ', doc.text[max(pred['pos'][0] - 10, 0): pred['pos'][1] + 10])
            continue

        doc_labs[lab_span.start] = 'B-' + pred['name']
        for i in range(lab_span.start + 1, lab_span.end):
            doc_labs[i] = 'I-' + pred['name']

    return doc_labs, num_none


def convert_to_bi(nlp, texts, lab_preds, labels_to_use: List[str] = ['ORG', 'MISC', 'LOC', 'PER']):
    docs = list(nlp.pipe(texts))
    bi_preds = []
    for i, doc in enumerate(docs):
        doc_labs, _ = spans_to_bi_one_doc(doc, lab_preds[i])
        bi_preds.extend(doc_labs)

    return bi_preds
  

def build_graph(matrix):
    graph = defaultdict(list)

    for idx in range(0, len(matrix)):
        for col in range(idx+1, len(matrix)):
            graph[idx].append((col, matrix[idx][col] ))
    return graph

def BFS(s, end, graph, max_size=-1, black_list_relation=[]):
    """

    args:
        s - head index
        end - tail index
        graph - compressed attn graph
        max_size - using all the tokens
        black_list_relation - noun chunk indices

    returns:
        candidate_facts
    """
    # print('compressed attn graph')
    # print(graph)

    visited = [False] * (max(graph.keys())+100)

    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it
    queue.append((s, [(s, 0)]))

    found_paths = []

    visited[s] = True

    while queue:

        s, path = queue.pop(0)

        # Get all adjacent vertices of the
        # dequeued vertex s. If an adjacent
        # has not been visited, then mark it
        # visited and enqueue it
        for i, conf in graph[s]:
            if i == end:
                found_paths.append(path+[(i, conf)])
                break
            if visited[i] == False:
                queue.append((i, copy(path)+[(i, conf)]))
                visited[i] = True

    candidate_facts = []
    for path_pairs in found_paths:
        if len(path_pairs) < 3:
            continue
        path = []
        cum_conf = 0
        for (node, conf) in path_pairs:
            path.append(node)
            cum_conf += conf  # adding confidence scores together

        if path[1] in black_list_relation:
            continue

        candidate_facts.append((path, cum_conf))

    candidate_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)
    # print('candidate facts:')
    # print(candidate_facts)
    return candidate_facts


def get_intermediate(graph, head_idx, tail_idx, count):
    """
    helper function for beam_search to return the common tokens that have high attention

    args:
        graph - attention graph
        head_idx (Integer) - index of the head entity
        tail_idx (integer) - index of the tail entity
        count (integer) - number of tokens

    returns:
        candidate_facts (List) - list of candidate facts with conf scores
    """

    candidate_relations = list()
    for i, conf in graph[head_idx]:
        if i > head_idx and i < tail_idx:
            candidate_relations.append((i, conf))

    # create 3 tuples (single token relation)
    # this code could cause mistakes if indexed wrong: graph[i][tail_idx][1]
    tups = [([head_idx, i, tail_idx], 0 + conf + graph[i][tail_idx-i-1][1]) for i, conf in candidate_relations]
    tups.sort(key= lambda t: -t[1])

    return tups[:count]



def beam_search(s, end, graph, max_size=-1, black_list_relation=[], k=3):
    """
    modified version of BFS to account for beams

    args:
        s - head index
        end - tail index
        graph - compressed attn graph
        max_size - using all the tokens
        black_list_relation - noun chunk indices

    returns:
        candidate_facts
    """
    # print('compressed attn graph')
    # print(graph)
    head_idx = s

    visited = [False] * (max(graph.keys())+100)

    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it
    queue.append((s, [(s, 0)]))

    found_paths = []

    visited[s] = True

    while queue:

        s, path = queue.pop(0)

        # Get all adjacent vertices of the
        # dequeued vertex s. If an adjacent
        # has not been visited, then mark it
        # visited and enqueue it
        agg_conf = path[-1][1]
        for i, conf in graph[s]:
            if i == end:
                # keep aggregate conf
                found_paths.append(path+[(i, conf + agg_conf)])
                break
            else:
                queue.append((i, copy(path)+[(i, conf + agg_conf)]))

        # narrow down queue
        # indexing for sort, tuple-list-tuple
        sorted_queue = sorted(queue, key=lambda t: -t[1][-1][1])

        # select top k
        queue = sorted_queue[:k]

    candidate_facts = []
    for path_pairs in found_paths:
        if len(path_pairs) < 3:
            continue
        path = []
        cum_conf = 0
        for (node, conf) in path_pairs:
            path.append(node)
            cum_conf = conf  # don't need to add confidence scores together

        if path[1] in black_list_relation:
            continue

        candidate_facts.append((path, cum_conf))


    candidate_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)
#     print('candidate facts:', candidate_facts)

    # append top 3 single word indices with head that are less than end idx
    try:
        top3 = get_intermediate(graph, head_idx, end, 3)
    except Exception as e:
        print('unable to extract top3 intermediate words')
        top3 = []

#     print('top3:', top3)
    candidate_facts += top3

    # print('candidate facts:')
    # print(candidate_facts)
    return candidate_facts


def is_word(token):
    if len(token) == 1 and alphabet.match(token) == None:
        return False
    return True

def create_mapping(sentence, return_pt=False, nlp = None, tokenizer=None, debug=False):
    '''Create a mapping
        nlp: spacy model
        tokenizer: huggingface tokenizer
    '''
    doc = nlp(sentence)

    tokens = list(doc)

    # chunk2id = {}

    ent2type = dict()

    start_ents = []
    end_ents = []
    ents = []

    start_chunk = []
    end_chunk = []
    noun_chunks = []

    start_ent_nc = []
    end_ent_nc = []
    ent_nc = []

    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk.text)
        start_chunk.append(chunk.start)
        end_chunk.append(chunk.end)

    for ent in doc.ents:  # sb: use entities instead of non chunks
        ents.append(ent.text)
        start_ents.append(ent.start)
        end_ents.append(ent.end)
        ent2type[ent.text] = ent.label_

    # incorporate noun chunks into the entities
    # i = 0  # noun chunk idx
    # j = 0  # entity idx
    # num_nc = len(noun_chunks)
    # num_ent = len(ents)
    # while i < num_nc or j < num_ent:
    #     if i == num_nc:
    #         ent_nc += ents[j:]
    #         start_ent_nc += start_ents[j:]
    #         end_ent_nc += end_ents[j:]
    #         break
    #     if j == num_ent:
    #         ent_nc += noun_chunks[i:]
    #         start_ent_nc += start_chunk[i:]
    #         end_ent_nc += end_chunk[i:]
    #         break

    #     # noun chunk starts first
    #     if ((start_chunk[i] <= start_ents[j]) and (end_chunk[i] >= start_ents[j])) or \
    #         ((start_chunk[i] <= end_ents[j]) and (end_chunk[i] >= end_ents[j])):
    #         # overlap case
    #         start_ent_nc.append(start_chunk[i])
    #         end_ent_nc.append(end_chunk[i])
    #         ent_nc.append(noun_chunks[i])
    #         i += 1
    #         j += 1
    #     elif end_chunk[i] <= start_ents[j]:
    #         start_ent_nc.append(start_chunk[i])
    #         end_ent_nc.append(end_chunk[i])
    #         ent_nc.append(noun_chunks[i])
    #         i += 1
    #     elif start_chunk[i] >= end_ents[j]:
    #         # take entity j
    #         start_ent_nc.append(start_ents[j])
    #         end_ent_nc.append(end_ents[j])
    #         ent_nc.append(ents[j])
    #         j += 1
    #     else:
    #         print('Combining ents and ncs - this case should never be encountered.')

    # reassign values for simplicity
    noun_chunks = ents
    start_chunk = start_ents
    end_chunk = end_ents

    sentence_mapping = []  # compressed sentence (noun chunk is a single string)
    token2id = {}  # given a noun chunk, get the index in the compressed sentence
    mode = 0 # 1 in chunk, 0 not in chunk
    chunk_id = 0
    for idx, token in enumerate(doc):
        if idx in start_chunk:
            mode = 1
            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = len(token2id)
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token.text)
            token2id[sentence_mapping[-1]] = len(token2id)
    if debug: print('sentence mapping:', sentence_mapping)

    # sb: create chunks of text
    ents_posn = [(e.start, e.end) for e in doc.ents]
    curr_ent = 0
    text2posn = dict()
    posn2text = dict()
    text_list = list()
    idx_list = list()
    idx2posn = dict()

    tokens_text = [t.text for t in tokens]

    # sb: everything before and after ents_posn are treated as 1 to 1
    curr = 0
    for start, end in ents_posn:
        if start > curr:
            text_list += tokens_text[curr:start]
            idx_list += [[idx] for idx in range(curr, start)]
        text_list.append(' '.join(tokens_text[start:end]))
        idx_list.append(list(range(start, end)))
        curr = end
    if curr < len(tokens_text):
        text_list += tokens_text[curr:]
        idx_list += [[idx] for idx in range(curr,len(tokens_text))]

    for k, lst_idx in enumerate(idx_list):
        for idx in lst_idx:
            idx2posn[idx] = k

    text2posn = {t:i for i, t in enumerate(text_list)}
    posn2text = {v:k for k, v in text2posn.items()}

    token_ids = []
    tokenid2word_mapping = []
    token2subtoken_ids = list()

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token), add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [ token2id[token] ]*len(subtoken_ids)
        token_ids += subtoken_ids
        if debug: print(f'token: {token}, subtoken_ids: {subtoken_ids}')

    for token in doc:
        subtoken_ids = tokenizer(str(token.text), add_special_tokens=False)['input_ids']
        token2subtoken_ids.append(subtoken_ids)

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)

    return outputs, tokenid2word_mapping, token2id, noun_chunks, token2subtoken_ids, ent2type

def compress_attention(attention, tokenid2word_mapping, operator=np.mean):

    new_index = []

    prev = -1
    for idx, row in enumerate(attention):
        try:
            token_id = tokenid2word_mapping[idx]  # assume this returns an index
        except Exception as e:
            print(f'idx: {idx}')
            print('len tokenid2word:', len(tokenid2word_mapping))
            raise Exception(e)

        if token_id != prev:
            # when reaching a word that maps to a new chunk, append new set of attention
            new_index.append( [row])
            prev = token_id
        else:
            # add to the most recent batch of attentions
            # we will average over these to compress it
            new_index[-1].append(row)

    new_matrix = []

    # compress attention vertically by taking the average across tokens
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

#     print('new_index matrix:', new_index)
#     print('len of new_index matrix:', len(new_index))
#     print('new_matrix matrix:', new_matrix)
#     print('shape of new_matrix:', new_matrix.shape)

    # transpose so we can compress in the other direction
    attention = np.array(new_matrix).T

    prev = -1
    new_index=  []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)


    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    return new_matrix.T


def index2word(tokenid2word_mapping, token2id):
    tokens = []
    prev = -1
    for token_id in tokenid2word_mapping:
        if token_id == prev:
            continue

        tokens.append(token2id[token_id])
        prev = token_id

    return tokens

# def get_pps(doc):
#     "Function to get PPs from a parsed document."
#     pps = []
#     for token in doc:
#         # Try this with other parts of speech for different subtrees.
#         if token.pos_ == 'ADP':
#             pp = ' '.join([tok.orth_ for tok in token.subtree])
#             pps.append(pp)
#     return pps


if __name__ == '__main__':
    import en_core_web_sm
    from transformers import AutoTokenizer, BertModel
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    encoder = BertModel.from_pretrained('bert-base-cased')
    nlp = en_core_web_sm.load()

    sentence = 'Rolling Stone wrote: “No other pop song has so thoroughly challenged artistic conventions”'
    sentence = 'Dylan sing "Time They Are Changing"'
    inputs, tokenid2word_mapping, token2id, noun_chunks  = create_mapping(sentence, return_pt=True, nlp=nlp, tokenizer=tokenizer)

    outputs = encoder(**inputs, output_attentions=True)
    print(noun_chunks, tokenid2word_mapping, token2id)
