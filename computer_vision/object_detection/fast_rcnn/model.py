import torch
from torch import nn
from torchvision import models
from torchvision.ops import RoIPool

class FastRCNN(nn.Module):
    def __init__(self, num_classes, spatial_scale=1.0/16.0):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        self.roi_pool = RoIPool((7, 7), spatial_scale)
        feature_dim = 512 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.classifier = nn.Linear(1024, num_classes + 1) 
        self.regressor = nn.Linear(1024, num_classes * 4)

    def forward(self, images, rois):
        feature_maps = self.feature_extractor(images)
        pooled_features = self.roi_pool(feature_maps, rois)
        flattened = torch.flatten(pooled_features, start_dim=1)
        shared_fc = self.fc(flattened)

        class_logits = self.classifier(shared_fc)
        bbox_offsets = self.regressor(shared_fc)
        
        return class_logits, bbox_offsets