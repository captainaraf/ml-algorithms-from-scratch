import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, feature_map):
        t = self.relu(self.conv(feature_map))
        
        cls_logits = self.cls_head(t) 
        bbox_offsets = self.reg_head(t)
        
        return cls_logits, bbox_offsets
    
class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        out_channels = 512 
        
        self.rpn = RegionProposalNetwork(in_channels=out_channels, num_anchors=9)
        
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['0'], 
            output_size=7, 
            sampling_ratio=2
        )
        
        feature_dim = out_channels * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(1024, num_classes + 1)
        self.regressor = nn.Linear(1024, num_classes * 4)

    def forward(self, images, gt_boxes=None):
        batch_size = images.size(0)
        
        feature_maps = self.backbone(images) 
        
        rpn_cls_logits, rpn_bbox_offsets = self.rpn(feature_maps)
        
        proposals = self._generate_mock_proposals_for_flow(feature_maps, batch_size)
        features_dict = {'0': feature_maps}
        pooled_features = self.roi_align(features_dict, proposals, [images.shape[-2:]])
        
        flattened = torch.flatten(pooled_features, 1)
        shared_fc = self.fc(flattened)
        
        class_logits = self.classifier(shared_fc)
        bbox_offsets = self.regressor(shared_fc)
        
        return rpn_cls_logits, rpn_bbox_offsets, class_logits, bbox_offsets

    def _generate_mock_proposals_for_flow(self, feature_maps, batch_size):
        _, _, feat_h, feat_w = feature_maps.shape
        proposals = []
        for i in range(batch_size):
            dummy_roi = torch.tensor([
                [float(i), 10.0, 10.0, 100.0, 100.0],
                [float(i), 50.0, 30.0, 200.0, 180.0]
            ], device=feature_maps.device)
            proposals.append(dummy_roi)
        return torch.cat(proposals, dim=0)