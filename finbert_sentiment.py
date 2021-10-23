
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F

# Model is from: https://huggingface.co/ProsusAI/finbert
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
classes = ['positive', 'negative', 'neutral']

def classify(sentences, **kwargs):
    
    out = tokenizer(sentences, padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        preds = model(input_ids=torch.tensor(out['input_ids']), attention_mask=torch.tensor(out['attention_mask']))
        preds = F.softmax(preds.logits, dim=1)

    out = []
    for p in preds:
        out.append({k:p[i].item() for i,k in enumerate(classes)})
    
    return out


if __name__=='__main__':
    sentences = ['Stocks rallied and the British pound gained.', "There wasn't much change in the market today.", "Telsa's stock price plummeted."]
    
    out = classify(sentences)
    print(out)
