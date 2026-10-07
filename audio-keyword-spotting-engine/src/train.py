"""
Training loop tying AudioPreprocessor + AudioCNNClassifier together.

NOTE: no real audio dataset ships with this repo. To keep this
runnable with zero setup (no network, no audio files), training runs
on synthetic sine-wave-based waveforms standing in for different
keyword classes, for a few epochs - so the full pipeline (waveform ->
mel-spectrogram -> CNN -> loss -> backward) runs and is verified end
to end. Swap `make_synthetic_batch()` for a real DataLoader over
recorded `.wav` keyword clips for production use.
"""
import torch
import torch.nn as nn

from audio_processor import AudioPreprocessor
from model import AudioCNNClassifier


def make_synthetic_batch(batch_size=8, sample_rate=16000, duration_s=1.0, num_classes=5):
    """Each class gets a distinct sine frequency + noise, so there's at
    least some learnable signal rather than pure random noise."""
    t = torch.linspace(0, duration_s, int(sample_rate * duration_s))
    labels = torch.randint(0, num_classes, (batch_size,))
    waveforms = torch.stack([
        torch.sin(2 * torch.pi * (200 + 100 * label.item()) * t) + 0.05 * torch.randn_like(t)
        for label in labels
    ]).unsqueeze(1)  # (batch, 1, samples)
    return waveforms, labels


def train(epochs=5, batches_per_epoch=10, num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = AudioPreprocessor()
    model = AudioCNNClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            waveforms, labels = make_synthetic_batch(num_classes=num_classes)
            waveforms, labels = waveforms.to(device), labels.to(device)

            specs = preprocessor.process(waveforms)  # (batch, 1, n_mels, time)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {epoch_loss / batches_per_epoch:.4f}")

    torch.save(model.state_dict(), "keyword_spotter.pt")
    print("Saved checkpoint to keyword_spotter.pt")


if __name__ == "__main__":
    train()
