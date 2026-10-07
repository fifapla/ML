# Fine-Tuned Transformer for Sentiment Analysis
Fine-tuning DistilBERT using PyTorch for multi-class sentiment classification.


**Note:** src/train.py was added (previously missing) with a tiny 6-example toy dataset so the full tokenize -> train -> save pipeline is complete and runs end to end. Needs internet access on first run to download the pretrained DistilBERT weights/tokenizer - will not run fully offline. Swap the toy dataset for a real labeled dataset (e.g. via the datasets library) for anything beyond a pipeline smoke test.
