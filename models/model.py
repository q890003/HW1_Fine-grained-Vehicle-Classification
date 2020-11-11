import torch
import torch.nn as nn
from torchvision import models


def get_resnet101_32x8d_pretrained_model(numClasses=196):
    model_ft = models.resnext101_32x8d(pretrained=True)
    numFtrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(numFtrs, numClasses)
    return model_ft
