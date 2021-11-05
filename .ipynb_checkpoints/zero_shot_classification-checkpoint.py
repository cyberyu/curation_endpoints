

from transformers import pipeline

"""
Is defined https://huggingface.co/facebook/bart-large-mnli and can also handle case where multiple
true labels
"""

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

def classify(sentences, label_names):
    if type(sentences) is str:
        sentences = [sentences]
    
    label_names_to_use = [x.replace('_', ' ') for x in label_names]
    label_mapping = {k:v for k,v in zip(label_names_to_use, label_names)}
    output = classifier(sentences, label_names_to_use) #candidate_labels)
    print(output)
    # convert to list of dictionaries of label_names to 
    out = []
    for el in output:
        out.append({label_mapping[lab]:el['scores'][i] for i,lab in enumerate(el['labels'])})
    
    return out

    
if __name__ == '__main__':
    sequences_to_classify = ["one day I will see the world", "I love making pasta with my grandma", "I'm making pasta today."]
    #sequence_to_classify = [sequence_to_classify]*100
    candidate_labels = ['travel', 'cooking', 'dancing']
    out = classify(sequences_to_classify, candidate_labels) #classifier(sequence_to_classify, candidate_labels)
    print(out)
    
    print()
    candidate_labels = ['positive', 'negative']
    sequences_to_classify = ["I'm not happy with the outcome", "why do you always have to do that.", "I can't believe how nice the day is."]
    print(classify(sequences_to_classify, candidate_labels))
    #{'labels': ['travel', 'dancing', 'cooking'],
    # 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
    # 'sequence': 'one day I will see the world'}
