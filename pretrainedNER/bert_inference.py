import torch
import numpy as np
import time


def get_preds(texts, tokenizer, model, nlp, device):
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    ind_to_label = {i: k for i, k in enumerate(label_list)}

    if not type(texts) is list:
        texts = [texts]

    all_ner_preds = []
    docs = nlp.pipe(texts)
    for doc in docs:
        stime = time.time()

        sentences = []
        sent_start_and_ends = []
        for sent_ind, sent in enumerate(doc.sents):
            # if sentence is too long, then split into smaller ones. Should be very rare

            for s_ind in range(0, len(sent), 150):
                sentences.append([x.text for x in sent[s_ind:s_ind + 150]])
                chunk_start = sent.start + s_ind
                sent_start_and_ends.append((chunk_start, min(sent.end, chunk_start + 150)))

        # break up into sentences so we don't have to truncate
        # sentences = [[x.text for x in sent] for sent in doc.sents]
        tokenized_inputs = tokenizer(
            sentences,
            padding=True,  # For now just single sentence encoded at a time
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_tensors='pt'
        ).to(device)

        # process in batches so no OOM error
        output_logits = None
        bsz = 32
        for i in range(0, len(sentences), bsz):

            with torch.no_grad():
                model_outputs = model(input_ids=tokenized_inputs['input_ids'][i:i + bsz],
                                      attention_mask=tokenized_inputs['attention_mask'][
                                                     i:i + bsz])  # **tokenized_inputs)

            if output_logits is None:
                output_logits = model_outputs.logits.cpu()
            else:
                output_logits = torch.cat([output_logits, model_outputs.logits.cpu()], dim=0)
                print('HERE')

        print(output_logits.shape)

        predictions = torch.argmax(output_logits, dim=2)
        probs = torch.softmax(output_logits, dim=2)

        print('Time1: ', time.time() - stime)
        stime = time.time()

        # pred_labels = [] # Get pred_labels for debugging purposes.
        ner_preds = []
        for sent_ind, sent in enumerate(sentences):
            cur_start_ind = -1
            cur_label = 'O'
            confidences = []
            sent_start, sent_end = sent_start_and_ends[sent_ind]
            for i in range(len(sent)):
                # for i in range(len(tokens)):

                tok_ind = tokenized_inputs.word_to_tokens(sent_ind, i).start
                pred = predictions[sent_ind, tok_ind].item()
                pred_prob = probs[sent_ind, tok_ind, pred].item()
                confidences.append(pred_prob)
                # pred_labels.append(ind_to_label[pred])
                pred_str = label_list[pred][2:] if label_list[pred] != 'O' else 'O'

                if pred_str != cur_label:

                    if cur_label != 'O':
                        # ended an entity span
                        ner_preds.append({'name': cur_label,
                                          'tpos': [sent_start + cur_start_ind, sent_start + i],
                                          'confidence': np.mean(confidences[cur_start_ind:i])})
                        cur_label = 'O'
                        cur_start_ind = -1

                    if pred_str != 'O':
                        # then started new ent
                        cur_label = pred_str
                        cur_start_ind = i

            # handle case of entity at end of question
            if cur_start_ind != -1:
                ner_preds.append({'name': cur_label,
                                  'tpos': [sent_start + cur_start_ind, sent_end],
                                  'confidence': np.mean(confidences[cur_start_ind:])})

        # touch up dictionary to create format AnnotationTool expects
        for ent in ner_preds:
            span_last_tok = doc[ent['tpos'][1] - 1]
            span_first_tok = doc[ent['tpos'][0]]
            ent['pos'] = [span_first_tok.idx, span_last_tok.idx + len(span_last_tok) - 1]
            ent['text'] = doc[ent['tpos'][0]:ent['tpos'][1]].text
            ent['tpos'][1] -= 1  # assume that we include last index

        all_ner_preds.append(ner_preds)
        print('Time2: ', time.time() - stime)
    return all_ner_preds