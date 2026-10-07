"""
Training loop for the LSTM forecaster.

NOTE: no real price series ships with this repo. To keep this runnable
with zero setup, training runs on synthetic random sequences for a few
epochs so the loop itself is real and verified (loss is actually
computed and backpropagated each step) - swap `make_synthetic_batch()`
for a real windowed time-series DataLoader for production use.
"""
import os
import torch
import torch.nn as nn
from model import LSTMForecaster


def make_synthetic_batch(batch_size=32, seq_len=10):
    # Each sequence is a random walk, so there's at least *some* real
    # temporal structure for the LSTM to pick up on, rather than pure
    # i.i.d. noise with no signal at all.
    steps = torch.randn(batch_size, seq_len, 1) * 0.1
    series = torch.cumsum(steps, dim=1)
    target = series[:, -1, :] + torch.randn(batch_size, 1) * 0.05
    return series, target


def train(epochs=20, batches_per_epoch=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            inputs, targets = make_synthetic_batch()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {epoch_loss / batches_per_epoch:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_forecaster.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    train()
