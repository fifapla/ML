# Industrial Defect Detection using PyTorch
Deep Learning pipeline using Transfer Learning (ResNet18) for automated defect classification.


**Note:** train.py now runs a real multi-epoch training loop (forward/backward/optimizer step each batch), but on synthetic random tensors since no real defect images ship with this repo. pretrained=False is used so it runs without network access; swap to pretrained=True (needs internet, downloads ImageNet weights) and a real DataLoader over your own labeled images for production use.
