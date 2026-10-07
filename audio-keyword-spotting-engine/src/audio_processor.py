import torch
import torchaudio
import torchaudio.transforms as T

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=64):
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )

    def process(self, waveform):
        spectrogram = self.mel_spectrogram(waveform)
        log_spectrogram = torch.log(spectrogram + 1e-9)
        return log_spectrogram
