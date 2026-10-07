import torch.nn as nn
from torchvision import models

def get_defect_model(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model
