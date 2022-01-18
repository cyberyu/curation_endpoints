from utils import compress_attention, create_mapping, BFS, build_graph, is_word, beam_search
from multiprocessing import Pool
import spacy
import en_core_web_md
import torch
from transformers import AutoTokenizer, BertModel, GPT2Model
from constant.constant import invalid_relations_set, prepositions
from spacy.matcher import Matcher
# from spacy.util import filter_spans
def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result

from collections import Counter

import numpy as np  # sb

pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'AUX', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]

# instantiate a Matcher instance


def process_matrix(attentions, layer_idx = -1, head_num = 0, avg_head=False, trim=True, use_cuda=True):
    if avg_head:
        if use_cuda:
            attn =  torch.mean(attentions[0][layer_idx], 0).cpu()
        else:
            attn = torch.mean(attentions[0][layer_idx], 0)
        attention_matrix = attn.detach().numpy()
    else:
        attn = attentions[0][layer_idx][head_num]
        if use_cuda:
            attn = attn.cpu()
        attention_matrix = attn.detach().numpy()

    attention_matrix = attention_matrix[1:-1, 1:-1]  # to get rid of added start/end tokens

    return attention_matrix

def bfs(args):
    s, end, graph, max_size, black_list_relation = args
    # return BFS(s, end, graph, max_size, black_list_relation)
    return beam_search(s, end, graph, max_size, black_list_relation, k=6)


def check_relations_validity(relations):
    for rel in relations:
        if rel.lower() in invalid_relations_set or rel.isnumeric():
            return False
    return True

def global_initializer(nlp_object):
    global spacy_nlp
    spacy_nlp = nlp_object

def filter_relation_sets(params):
    triplet, id2token, debug = params

    triplet_idx = triplet[0]
    confidence = triplet[1]
    head, tail = triplet_idx[0], triplet_idx[-1]
#     print('head:',head)
#     print('tail:',tail)
#     print('id2token:',id2token)
    if head in id2token and tail in id2token:
        head = id2token[head]
        tail = id2token[tail]
        relations = [ spacy_nlp(id2token[idx])[0].lemma_  for idx in triplet_idx[1:-1] if idx in id2token ]
        # relations = [ id2token[idx] for idx in triplet_idx[1:-1] if idx in id2token ]
#         print(f'head: {head}, tail: {tail}, relations: {relations}')
#         if debug:
#             print('relations:', relations)
#             print('check_relations_validity(relations):', check_relations_validity(relations))
#             print('head.lower() not in invalid_relations_set:', head.lower() not in invalid_relations_set)
#             print('tail.lower() not in invalid_relations_set:', tail.lower() not in invalid_relations_set)
        if len(relations) > 0 and check_relations_validity(relations) and head.lower() not in invalid_relations_set and tail.lower() not in invalid_relations_set:
            return {'h': head, 't': tail, 'r': relations, 'c': confidence }
    return {}


def pathToNode(root, path, k):
#     print(f'root: {root}, path: {path}, k: {k}')
    # base case handling
    if root is None:
        return False

     # append the node value in path
    path.append(root.i)

    # See if the k is same as root's data
    if root.i == k :
#         print('root.i == k')
        return True

    # Check if k is found in left or right
    # sub-tree
    for child in root.children:
        if pathToNode(child, path, k):
            return True

    path.pop()
    return False


def distance(root, data1, data2):
#     print(f'root: {root}, data1: {data1}, data2: {data2}')
    if root:
        # store path corresponding to node: data1
        path1 = []
        pathToNode(root, path1, data1)

        # store path corresponding to node: data2
        path2 = []
        pathToNode(root, path2, data2)

        # iterate through the paths to find the
        # common path length
        i=0
        while i<len(path1) and i<len(path2):
            # get out as soon as the path differs
            # or any path's length get exhausted
            if path1[i] != path2[i]:
                break
            i = i+1

        # get the path length by deducting the
        # intersecting path length (or till LCA)
        return (len(path1)+len(path2)-2*i)
    else:
        return 0


def calculate_dependency_matrix(doc, normalizing_strength=1):
    """
    calculate the dependency matrix for doc
    """
    all_rows = list()

    for sent in doc.sents:
        for i, tok1 in enumerate(sent):
            row = list()
            for j, tok2 in enumerate(sent):
                d = distance(sent.root, tok1.i, tok2.i)
                row.append(d)
            all_rows.append(row)

    distances = np.array(all_rows)
    dependency_matrix = np.exp(- distances / normalizing_strength)

    return dependency_matrix


def filter_triplets(ent_filter, rel_filter, triplets):
    """
    filter triplets based on entity filter and relation filter
    """

    new_triplets = list()

    for trip in triplets:
        # make sure each word of entities in head, tail are in ent_filter
        new_head = ' '.join([he for he in trip['h'].split() if he in ent_filter]).strip()
        new_tail = ' '.join([ta for ta in trip['t'].split() if ta in ent_filter]).strip()

        # make sure each word of relations are in rel_filter
        # new_rel = ' '.join([re for re in trip['r'][0].split() if re in rel_filter]).strip()
        new_rel = [re for re in trip['r'] if re in rel_filter]

        # if last word is a preposition and verb selected, keep preposition
        if trip['r'][-1] in prepositions and len(new_rel) > 0 and trip['r'][-1] not in new_rel:
            new_rel.append(trip['r'][-1])

        # remove triplets that have head, relation, or tail empty string
        if new_head == '' or new_tail == '' or len(new_rel) == 0:
            continue
        else:
            new_trip = trip.copy()
            new_trip['r'] = new_rel
            new_trip['h'] = new_head
            new_trip['t'] = new_tail
            new_triplets.append(new_trip)

    return new_triplets



def add_ent_types(triplets, ent2type):
    """add the entity types to the triplets

    Arguments:
        triplets {List} -- list of Dictionary of h, r, t, conf
        ent2type {Dict} -- Mapping from entity to entity type

    returns:
        new_triplets {List} -- dictionary of augmented triplets
    """
    new_triplets = list()

    for trip in triplets:
        trip['h_type'] = ent2type.get(trip['h'], '')
        trip['t_type'] = ent2type.get(trip['t'], '')
        new_triplets.append(trip)

    return new_triplets


# SHi Yu 2022-01-16 ---------------edit start
def add_token_index(triplets, chunk2pos, charidx, r_lemma_lookup):

    new_triplets=list()

    for trip in triplets:

        htpos=chunk2pos[trip['h']]
        ttpos=chunk2pos[trip['t']]

        r_original=r_lemma_lookup[trip['r'][0]]

        if len(r_original)>1:
            r_original=r_original[0]
        else:
            r_original=r_original[0]

        rtpos = chunk2pos[r_original]


        hpos = charidx[trip['h']]
        tpos = charidx[trip['t']]
        rpos = charidx[r_original]
        
        mindistance=1000
        for i in htpos:
            for j in ttpos:
                for k in rtpos:
                    if max(i[-1],j[-1],k[-1])-min(i[0],j[0],k[0])<mindistance:
                        bestcomb_tpos = [i,j,k]
                        mindistance = max(i[-1],j[-1],k[-1])-min(i[0],j[0],k[0])
                        
        mindistance=1000
        for i in hpos:
            for j in tpos:
                for k in rpos:
                    if max(i[-1],j[-1],k[-1])-min(i[0],j[0],k[0])<mindistance:
                        bestcomb_pos = [i,j,k]
                        mindistance = max(i[-1],j[-1],k[-1])-min(i[0],j[0],k[0])
        
        trip['h_tpos']=[bestcomb_tpos[0][0],bestcomb_tpos[0][-1]]
        trip['t_tpos']=[bestcomb_tpos[1][0],bestcomb_tpos[1][-1]]
        trip['r_tpos']=[bestcomb_tpos[2][0],bestcomb_tpos[2][-1]]
        
        trip['h_pos']=[bestcomb_pos[0][0],bestcomb_pos[0][-1]]
        trip['t_pos']=[bestcomb_pos[1][0],bestcomb_pos[1][-1]]
        trip['r_pos']=[bestcomb_pos[2][0],bestcomb_pos[2][-1]]

        
        print('bestcomb_tpos ' + str(bestcomb_tpos) )
        print('bestcomb_pos ' + str(bestcomb_pos) )


        new_triplets.append(trip)

    return triplets
#Shi Yu 2022-01-16 ------------edit end

def expand_matrix(mat, token2subtoken_ids):
    """
    expand the input matrix, dependency matrix, to account for subtoken usage

    args:
        mat (ndarray) - matrix of dependencies
        token2subtoken_ids (Dict) - dictionary that maps tokens to subtoken_ids

    returns:
        expand_mat (ndarray) - expanded matrix of dependencies
    """

    expanded_matrix = list()

    for i, row in enumerate(mat):
        expanded_matrix += [row] * len(token2subtoken_ids[i])

    temp = np.array(expanded_matrix)
    expanded_matrix = list()

    temp = np.transpose(temp)
    for i, row in enumerate(temp):
        expanded_matrix += [row] * len(token2subtoken_ids[i])

    expanded_matrix = np.array(expanded_matrix)

    return expanded_matrix



def parse_sentence(sentence, tokenizer, encoder, nlp, use_cuda=True, use_filter=True, tree_weight=0.0, debug=False):
    '''Implement the match part of MAMA

    '''
    tokenizer_name = str(tokenizer.__str__)

    #inputs, tokenid2word_mapping, token2id, noun_chunks, token2subtoken_ids, ent2type 
    # Shi Yu  2022-01-16  --edit start
    inputs, tokenid2word_mapping, token2id, noun_chunks, token2subtoken_ids, ent2type, idx2posn, posn2text, chunk2pos, charidx= create_mapping(sentence, return_pt=True, nlp=nlp, tokenizer=tokenizer, debug=debug)
    # Shi Yu  2022-01-16  --edit end


    # inputs are the tokenIds
    if debug:
        print('token2subtoken_ids:', token2subtoken_ids)
        print('len token2subtoken_ids:', len(token2subtoken_ids))
        print('inputs:', inputs)
        print('tokenid2word_mapping:', tokenid2word_mapping)

    doc = nlp(sentence)
    tokens = list(doc)

    # sb: added this
    matcher = Matcher(nlp.vocab)
    #matcher.add("Verb phrase", None, pattern)
    matcher.add("Verb phrase",[pattern], on_match=None)

    # sb: create a set for nouns and verbs
    entity_filter = set()  # words that can be used for entities
    relation_filter = set()  # words that can be used for relations

    for ent in doc.ents:
        entity_filter = entity_filter.union(set(ent.text.split(' ')))

    # call the matcher to find matches
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    relation_filter = set([a for s in spans for a in [e.lemma_ for e in list(s)]])

    #Shi Yu 2022-01-16 --edit start
    relation_filter_raw = set([a for s in spans for a in [e for e in list(s)]])
    #Shi Yu 2022-01-16 --edit end

    #Shi Yu 2022-01-16 --edit start
    r_lemma_lookup={}
    
    for e in list(set(relation_filter_raw)):
        if e.lemma_ in r_lemma_lookup.keys():
            vof = r_lemma_lookup[e.lemma_]
            vof.append(str(e))
            r_lemma_lookup[e.lemma_]=vof
        else:
            r_lemma_lookup[e.lemma_]=[str(e)]
    #Shi Yu 2022-01-16 --edit end

    if debug:
        print('entity filter:', entity_filter)
        print('relation filter:', relation_filter)
        #print('r_lemma_lookup:'. str(r_lemma_lookup))

    with torch.no_grad():
        if use_cuda:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        outputs = encoder(**inputs, output_attentions=True)
    trim = True
    if 'GPT2' in tokenizer_name:
        trim  = False

    '''
    Use average of last layer attention : page 6, section 3.1.2
    '''
    attention = process_matrix(outputs[2], avg_head=True, trim=trim, use_cuda=use_cuda)

    try:
        # sb: add layer of nlp mapping
        parse_tree_weight = tree_weight
        dependency_layer = calculate_dependency_matrix(doc)

        # expand dependency layer
        expanded_dependency_layer = expand_matrix(dependency_layer, token2subtoken_ids)
        expanded_dependency_layer = expanded_dependency_layer / expanded_dependency_layer.sum(axis=0, keepdims=1)

        added_attention = (1 - parse_tree_weight) * attention + parse_tree_weight * expanded_dependency_layer
    except Exception as e:
        print('could not add parse tree to attention')
        added_attention = attention

    if 'GPT2' in tokenizer_name:
        attention = attention.transpose(-1, -2)  # since GPT-2 is backward looking only, tr makes forward looking

#     merged_attention = compress_attention(attention, tokenid2word_mapping, operator=np.max)
    merged_attention = compress_attention(added_attention, tokenid2word_mapping, operator=np.max)

    attn_graph = build_graph(merged_attention)

#     print('attn graph:', attn_graph)

    tail_head_pairs = []
    for head in noun_chunks:
        for tail in noun_chunks:
            if head != tail:
                tail_head_pairs.append((token2id[head], token2id[tail]))

    black_list_relation = set([ token2id[n]  for n in noun_chunks ])

    all_relation_pairs = []
    id2token = { value: key for key, value in token2id.items()}

    # with Pool(10) as pool:
    with Pool(2) as pool:  # sb: 10 -> 2 to use less memory
        params = [  ( pair[0], pair[1], attn_graph, max(tokenid2word_mapping), black_list_relation, ) for pair in tail_head_pairs]
        for output in pool.imap_unordered(bfs, params):
#             print('bfs output:', output)
            if len(output):
                all_relation_pairs += [ (o, id2token, debug) for o in output ]
        pool.close()  # added by sb
        pool.join()  # added by sb


    triplet_text = []
    # with Pool(10, global_initializer, (nlp,)) as pool:
    # filter_relation_sets is where triples are lost
    with Pool(2, global_initializer, (nlp,)) as pool:  # sb: 10 -> 2 to use less memory
        for triplet in pool.imap_unordered(filter_relation_sets, all_relation_pairs):
            if len(triplet) > 0:
                triplet_text.append(triplet)

        pool.close()  # added by sb
        pool.join()  # added by sb

    if debug:
        print('triplet_text:', triplet_text)
        print('black_list_relations:', black_list_relation)
        # print('black_list_relations:', black_list_relation)

    # filter entity and relation words
    if use_filter: triplet_text = filter_triplets(entity_filter, relation_filter, triplet_text)

    # append entity types to tuples
    triplet_text = add_ent_types(triplet_text, ent2type)

    # Shi Yu 2022-01-16 added index
    triplet_text = add_token_index(triplet_text, chunk2pos, charidx, r_lemma_lookup)

    print('add token index ' + str(triplet_text))
    return triplet_text


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    nlp = en_core_web_md.load()
    selected_model = 'gpt2-medium'

    # use_cuda = True
    use_cuda = False


    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    encoder = GPT2Model.from_pretrained(selected_model)
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()

    target_file = [
        '../../Documents/KGERT-v2/datasets/squad_v1.1/train-v1.1.json',
        # '../../Documents/KGERT-v2/datasets/squad_v1.1/wiki_dev_2020-18.json',
        # '../../Documents/KGERT-v2/datasets/squad_v1/dev-v1.1.json',
    ]

    output_filename = [
        'train_v1.1.jsonl',
        # 'wiki_2020-18.jsonl',
        # 'dev-v1.1.jsonl',
    ]

    for target_file, output_filename in zip(target_file, output_filename):
        with open(target_file, 'r') as f:
            dataset = json.load(f)

        output_filename = selected_model +'_'+ output_filename

        print(target_file, output_filename)

        f = open(output_filename,'w')
        for data in tqdm(dataset['data'], dynamic_ncols=True):
            for para in data['paragraphs']:
                context = para['context']
                for sent in nlp(context).sents:
                    for output in parse_sentence(sent.text, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        f.write(json.dumps(output)+'\n')
                f.flush()

                for question in para['qas']:
                    question = question['question']
                    for output in parse_sentence(question, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        f.write(json.dumps(output)+'\n')
                f.flush()
        f.close()
