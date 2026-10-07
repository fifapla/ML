# Audio Keyword Spotting & Speech Classification
Speech recognition system converting raw audio waveforms to Mel-Spectrograms for CNN classification.


**Note:** src/train.py was added (previously missing - only the preprocessing and model classes existed) with synthetic sine-wave waveforms standing in for real keyword audio, so the full pipeline (waveform -> mel-spectrogram -> CNN -> loss -> backward) runs end to end. Swap make_synthetic_batch() for a real DataLoader over recorded .wav keyword clips for production use.
