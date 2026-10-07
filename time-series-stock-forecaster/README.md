# Time-Series Stock & Trend Forecaster
Deep Learning time-series prediction framework built with PyTorch and LSTM networks.


**Note:** train.py now runs a real 20-epoch training loop on synthetic random-walk sequences, since no real price data ships with this repo. Swap make_synthetic_batch() for a real windowed DataLoader over actual price history for production use.
