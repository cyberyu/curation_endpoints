# import spacy
# import pickle
# import time
# from spacy.tokens import Doc
# from typing import Dict, List, Any
# import copy
# import os
# import json
# import numpy as np

CONLL_TO_RETAIN = {"PER", "MISC", "ORG", "LOC"}

CONLL_MAPPINGS = {"PERSON": "PER", "COMPANY": "ORG", "GPE": "LOC", 'EVENT': "MISC", 'FAC': "MISC", 'LANGUAGE': "MISC",
                  'LAW': "MISC", 'NORP': "MISC", 'PRODUCT': "MISC", 'WORK_OF_ART': "MISC"}


# def to_docs(texts, labs, ignore_annotators=[], labels=[]):
#     unique_labs = set()
#     docs = list(nlp.pipe(texts))
#
#     # now need to add spangroups
#     for doc, labs_ in zip(docs, labs):
#         for annotator, span_list in labs_.items():
#             if any(x in annotator for x in
#                    ignore_annotators):  # ignore_annotators contains subsets of what we want to match
#                 continue
#
#             doc.spans[annotator] = []
#             for span in span_list:
#                 # if span['name'] not in labels:
#                 #    continue
#
#                 s = doc.char_span(span['pos'][0], span['pos'][1] + 1,
#                                   label=span['name'])  # span['pos'] includes final index so +1 here for slice indices
#                 if s is not None:
#                     doc.spans[annotator].append(s)
#                     unique_labs.add(span['name'])
#     # print(docs[0].spans)
#     return docs, unique_labs
#
#
# def from_docs(docs, ignore_annotators=[]):
#     docs_text = []
#     docs_labs = []
#     for doc in docs:
#         span_groups = doc.spans
#         docs_text.append(doc.text)
#         labs = {}
#         for annotator, span_list in span_groups.items():
#             if any(x in annotator for x in
#                    ignore_annotators):  # ignore_annotators contains subsets of what we want to match
#                 continue
#
#             # print(annotator)
#             # print(span_list)
#             labs[annotator] = [{'name': span.label_,
#                                 'pos': [span[0].idx, span[-1].idx + len(span[-1])],
#                                 'tpos': [span.start, span.end],
#                                 'text': span.text,
#                                 'confidence': 1
#                                 } for span in span_list]
#
#         docs_labs.append(labs)
#
#     return docs_text, docs_labs
#
#
# def extract_preds(docs, model_name):
#     all_preds = []
#     for doc in docs:
#         if model_name not in doc.spans:
#             all_preds.append([])
#             continue
#
#         # make sure indices for 'pos' and 'tpos' are such that they are inclusive of end index
#         all_preds.append([{'name': span.label_,
#                            'pos': [span[0].idx, span[-1].idx + len(span[-1]) - 1],
#                            'tpos': [span.start, span.end - 1],
#                            'text': span.text,
#                            'confidence': 1
#                            } for span in doc.spans[model_name]])
#
#     return all_preds
#
#
# def get_conll_data(save_folder, num_val_docs=10, num_test_docs=100):
#     # NOTE: these saved files have weak labels and true labels with spans that don't include the final index. The code for annotationTool expects
#     # that predicted spans have last index of span be inclusive. Therefore we alter that below.
#     # Assumes locations of the data files are known and does the loading / preprocessing
#     weak_labels_file = '/mnt/Data/labels.pkl'
#     true_labels_file = '/mnt/Data/true_labels.pkl'
#     texts_file = '/mnt/Data/texts.pkl'
#
#     with open(weak_labels_file, 'rb') as f:
#         weak_labels = pickle.load(f)
#
#     with open(true_labels_file, 'rb') as f:
#         true_labels = pickle.load(f)
#
#     with open(texts_file, 'rb') as f:
#         texts = pickle.load(f)
#
#     # print(weak_labels[0].keys())
#     # print(true_labels[0][:10])
#     # print(texts[0][:50])
#
#     # alter all true labels and weak labels by lowering 'pos' end indice by one
#     for d in true_labels:
#         for l in d:
#             l['pos'][1] -= 1
#             l['tpos'][1] -= 1
#
#     for d in weak_labels:
#         for k, preds in d.items():
#             for pred in preds:
#                 pred['pos'][1] -= 1
#                 pred['tpos'][1] -= 1
#
#     '''
#     if get_test_data:
#         # want to ignore final 'num_val_docs' docs when getting docs to test on since used in validation.
#         orig_len = len(texts)
#         inds_to_ignore = [len(texts)-i for i in range(1,num_val_docs+1)]
#         texts = [texts[i] for i in range(orig_len) if i not in inds_to_ignore]
#         labels = [weak_labels[i] for i in range(orig_len) if i not in inds_to_ignore]
#         true_labels = [true_labels[i] for i in range(orig_len) if i not in inds_to_ignore]
#         return texts, labels, true_labels
#     '''
#
#     # shuffle data
#     np.random.seed(1)  # so shuffle is same each time
#     ind_order = [i for i in range(len(true_labels))]
#     np.random.shuffle(ind_order)
#     weak_labels = [weak_labels[i] for i in ind_order]
#     true_labels = [true_labels[i] for i in ind_order]
#     texts = [texts[i] for i in ind_order]
#
#     data_dict = {}
#     data_dict['val_texts'] = texts[:num_val_docs]
#     data_dict['val_labels'] = weak_labels[:num_val_docs]
#     data_dict['val_true_labels'] = true_labels[:num_val_docs]
#
#     data_dict['test_texts'] = texts[num_val_docs:(num_val_docs + num_test_docs)]
#     data_dict['test_true_labels'] = true_labels[num_val_docs: (num_val_docs + num_test_docs)]
#
#     data_dict['train_texts'] = texts[(num_val_docs + num_test_docs):]
#     data_dict['train_labels'] = weak_labels[(num_val_docs + num_test_docs):]
#     data_dict['train_true_labels'] = true_labels[(num_val_docs + num_test_docs):]
#
#     print('Num tokens for val: ', sum(len(x.split(' ')) for x in data_dict['val_texts']))
#     print('Num tokens for test: ', sum(len(x.split(' ')) for x in data_dict['test_texts']))
#
#     # now fix up labels by adding gold standard in
#     # REmove this eventually. Right now keeps compatibility with DWS
#     for i, d in enumerate(data_dict['val_labels']):
#         d['gold_standard'] = data_dict['val_true_labels'][
#             i]  # doing this for now only for compatibility with other code
#
#     # NEED SOURCE_NAMES to use (Need to make sure always in same order)
#     # source_names = set()
#     # for d in weak_labels:
#     #    source_names = source_names.union(set(d.keys()))
#     # source_names = sorted([x for x in source_names if 'core_web' not in x])
#
#     source_names = ['date_detector', 'time_detector', 'money_detector', 'proper_detector', 'infrequent_proper_detector',
#                     'proper2_detector', 'infrequent_proper2_detector', 'nnp_detector', 'infrequent_nnp_detector',
#                     'compound_detector', 'infrequent_compound_detector', 'misc_detector', 'legal_detector',
#                     'company_type_detector', 'full_name_detector', 'number_detector', 'snips', 'core_web_md',
#                     'core_web_md_truecase', 'BTC', 'BTC_truecase', 'edited_BTC', 'edited_BTC_truecase',
#                     'edited_core_web_md', 'edited_core_web_md_truecase', 'wiki_cased', 'wiki_uncased',
#                     'multitoken_wiki_cased', 'multitoken_wiki_uncased', 'wiki_small_cased', 'wiki_small_uncased',
#                     'multitoken_wiki_small_cased', 'multitoken_wiki_small_uncased', 'geo_cased', 'geo_uncased',
#                     'multitoken_geo_cased', 'multitoken_geo_uncased', 'crunchbase_cased', 'crunchbase_uncased',
#                     'multitoken_crunchbase_cased', 'multitoken_crunchbase_uncased', 'products_cased',
#                     'products_uncased', 'multitoken_products_cased', 'multitoken_products_uncased', 'doclevel_voter',
#                     'doc_history_cased', 'doc_history_uncased', 'doc_majority_cased', 'doc_majority_uncased']
#     source_names = [x for x in source_names if ('core_web' not in x and 'doc_' not in x and 'doclevel' not in x)]
#     source_to_ind = {k: i for i, k in enumerate(source_names)}
#     label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
#
#     config = {'source_names': source_names,
#               'source_to_ind': source_to_ind,
#               'label_list': label_list}
#
#     if save_folder is not None:
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)
#
#         json.dump(config, open(os.path.join(save_folder, 'config.json'), 'w'))
#
#         for k, v in data_dict.items():
#             with open(os.path.join(save_folder, k + '.pkl'), 'wb') as f:
#                 pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     return config, data_dict
#
#
# def spans_to_bi_one_doc(doc: Doc, preds: List[Dict[str, Any]]) -> List[str]:
#     doc_labs = ['O'] * len(doc)
#     num_none = 0
#     for pred in preds:
#         # ex: {'name': 'ORG', 'pos': [0, 1], 'tpos': [0, 0], 'text': 'EU', 'confidence': 1}
#         lab_span = doc.char_span(pred['pos'][0], pred['pos'][1] + 1)  # last index is included in pred
#         if lab_span is None:
#             num_none += 1
#             if DEBUG:
#                 print('Pred resulted in span of none: ', doc.text[pred['pos'][0]:pred['pos'][1] + 1])
#                 print('region around span: ', doc.text[max(pred['pos'][0] - 10, 0): pred['pos'][1] + 10])
#             continue
#
#         doc_labs[lab_span.start] = 'B-' + pred['name']
#         for i in range(lab_span.start + 1, lab_span.end):
#             doc_labs[i] = 'I-' + pred['name']
#
#     return doc_labs, num_none
#
#
# def convert_to_bi(nlp, texts, lab_preds, labels_to_use: List[str] = ['ORG', 'MISC', 'LOC', 'PER']):
#     docs = list(nlp.pipe(texts))
#     bi_preds = []
#     for i, doc in enumerate(docs):
#         doc_labs, _ = spans_to_bi_one_doc(doc, lab_preds[i])
#         bi_preds.extend(doc_labs)
#
#     return bi_preds