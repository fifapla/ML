import torch
from transformers import AutoTokenizer
from src.model import SentimentClassifier

def predict_sentiment(text):
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SentimentClassifier(model_name=MODEL_NAME, num_classes=3)
    model.load_state_dict(torch.load('sentiment_model.pt'))
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    classes = ["Negative", "Neutral", "Positive"]
    return classes[pred_class], probs[0][pred_class].item()
