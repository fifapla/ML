"""
Training loop for the defect classifier.

NOTE: this repo ships no real image dataset (the data/ folder is a
placeholder). To keep this runnable with zero setup, this script trains
on a small batch of synthetic random tensors standing in for real
images, for a few epochs, so the training loop itself (forward,
loss, backward, optimizer step, checkpointing) is real and verified -
replace `make_synthetic_batch()` with a real DataLoader over
`DefectDataset` and your actual images for production use.
"""
import torch
import torch.nn as nn
from model import get_defect_model


def make_synthetic_batch(batch_size=16, num_classes=2):
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


def train(epochs=5, batches_per_epoch=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_defect_model(num_classes=2, pretrained=False).to(device)  # pretrained=False: no network access needed
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training initialized on device: {device}")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            images, labels = make_synthetic_batch()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {epoch_loss / batches_per_epoch:.4f}")

    torch.save(model.state_dict(), 'defect_resnet18.pth')
    print("Model checkpoint saved.")


if __name__ == '__main__':
    train()
