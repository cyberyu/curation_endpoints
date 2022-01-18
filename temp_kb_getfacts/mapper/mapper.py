from constant.constant import invalid_relations_set

# from REL.db.generic import GenericLookup
# sqlite_path = "../../Documents/wiki_2020/generated"
# sqlite_path = "../wiki_2014/wiki_2014/generated"
sqlite_path = "../wiki_2019/generated"
# emb = GenericLookup("entity_word_embedding", save_dir=sqlite_path, table_name="embeddings")



def Map(head, relations, tail, top_first=True, best_scores = True, debug=False):
    if head == None or tail == None or relations == None:
        return {}
    # head_p_e_m = emb.wiki(str(head), 'wiki')
    # if head_p_e_m is None:
    #     return {}
    # tail_p_e_m = emb.wiki(str(tail), 'wiki')
    # if tail_p_e_m is None:
    #     return {}
    # tail_p_e_m = tail_p_e_m[0]
    # head_p_e_m = head_p_e_m[0]
    head_p_e_m = [head]  # sb: use this as alternative to the entity linker
    tail_p_e_m = [tail]  # sb: use this as alternative to the entity linker

    # debug, print all boolean conditions
#     if debug:
#         print('Map function...')
#         for r in relations:
#             print('r not in invalid_relations_set:', r not in invalid_relations_set)
#             print('r.isalpha():', r.isalpha())
#             print('len(r) > 1:', len(r) > 1)
#             print('r:', r)


    valid_relations = [ r for r in relations if r not in invalid_relations_set and r.isalpha() and len(r) > 1 ]
    # valid_relations = relations

    if len(valid_relations) == 0:
        return {}

    return { 'h': head_p_e_m[0], 't': tail_p_e_m[0], 'r': '_'.join(valid_relations)  }
    # return { 'h': head_p_e_m[0], 't': tail_p_e_m[0], 'r': valid_relations  }


def deduplication(triplets):
    unique_pairs = []
    pair_confidence = []
    head_types = []
    tail_types = []
    h_tpos_list=[]
    h_pos_list=[]
    t_tpos_list=[]
    t_pos_list=[]
    r_tpos_list=[]
    r_pos_list=[]

    for t in triplets:
        key = '{}\t{}\t{}'.format(t['h'], t['r'], t['t'])
        conf = t['c']
        h_type = t['h_type']
        t_type = t['t_type']
        h_tpos=t['h_tpos']
        h_pos=t['h_pos']
        t_tpos=t['t_tpos']
        t_pos=t['t_pos']
        r_tpos=t['r_tpos']
        r_pos=t['r_pos']



        if key not in unique_pairs:
            unique_pairs.append(key)
            pair_confidence.append(conf)
            head_types.append(h_type)
            tail_types.append(t_type)
            h_tpos_list.append(h_tpos)
            h_pos_list.append(h_pos)
            t_tpos_list.append(t_tpos)
            t_pos_list.append(t_pos)
            r_tpos_list.append(r_tpos)
            r_pos_list.append(r_pos)


    unique_triplets = []
    for idx, unique_pair in enumerate(unique_pairs):
        h, r, t = unique_pair.split('\t')
        unique_triplets.append({ 'h': h, 'r': r, 't': t , 'c': pair_confidence[idx], 'h_type': head_types[idx], 't_type': tail_types[idx], 'h_tpos': h_tpos_list[idx], 'h_pos': h_pos_list[idx], 't_tpos': t_tpos_list[idx], 't_pos': t_pos_list[idx], 'r_tpos': r_tpos_list[idx], 'r_pos': r_pos_list[idx]})

    return unique_triplets


if __name__ == "__main__":
#     emb = GenericLookup("entity_word_embedding", save_dir=sqlite_path, table_name="embeddings")
#     p_e_m = emb.wiki("Bob", 'wiki')[:10]
#     print(p_e_m)
    print('removed main code')
