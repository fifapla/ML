"""
Fine-tuning loop for SentimentClassifier.

NOTE: this needs network access on first run to download the
pretrained DistilBERT weights/tokenizer from Hugging Face - it cannot
run in a fully offline/sandboxed environment. The toy dataset below is
a tiny hardcoded example so the training loop itself (tokenize -> batch
-> forward -> loss -> backward -> save) is complete and correct; swap
in a real labeled dataset (e.g. via `datasets.load_dataset`) for
anything beyond a pipeline smoke test.
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import TextDataset
from model import SentimentClassifier

TOY_TEXTS = [
    "This product is amazing, I love it!",
    "Terrible experience, would not recommend.",
    "It's okay, nothing special.",
    "Best purchase I've made all year.",
    "Completely broken on arrival, very disappointed.",
    "Average quality for the price.",
]
TOY_LABELS = [2, 0, 1, 2, 0, 1]  # 0=Negative, 1=Neutral, 2=Positive
CLASSES = ["Negative", "Neutral", "Positive"]


def train(epochs=3, batch_size=2, model_name="distilbert-base-uncased"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TextDataset(TOY_TEXTS, TOY_LABELS, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SentimentClassifier(model_name=model_name, num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "sentiment_model.pt")
    print("Saved checkpoint to sentiment_model.pt")


if __name__ == "__main__":
    train()
