import torch
from torch import nn
from torchvision import models

class RCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = self.feature_extractor[-1].out_features
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.regressor = nn.Linear(feature_dim, 4)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        class_logits = self.classifier(features)
        bbox_offsets = self.regressor(features)
        
        return class_logits, bbox_offsets