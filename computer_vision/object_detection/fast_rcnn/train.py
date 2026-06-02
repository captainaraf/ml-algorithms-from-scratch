import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import box_iou
import cv2
from torchvision import transforms

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.SmoothL1Loss()

    def forward(self, class_logits, bbox_offsets, gt_labels, gt_targets):
        loss_cls = self.cls_criterion(class_logits, gt_labels)
        pos_mask = gt_labels > 0
        if pos_mask.sum() > 0:
            num_classes = class_logits.shape[1] - 1
            bbox_offsets_reshaped = bbox_offsets.view(-1, num_classes, 4)
            
            assigned_classes = (gt_labels[pos_mask] - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, 4)
            pred_offsets = bbox_offsets_reshaped[pos_mask].gather(1, assigned_classes).squeeze(1)
            
            loss_bbox = self.bbox_criterion(pred_offsets, gt_targets[pos_mask])
        else:
            loss_bbox = torch.tensor(0.0, device=class_logits.device)
            
        return loss_cls + 1.0 * loss_bbox
    
def train_step(model, optimizer, criterion, image, gt_boxes, gt_labels):
    model.train()
    optimizer.zero_grad()
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    
    proposals = torch.tensor([[x, y, x+w, y+h] for (x, y, w, h) in rects[:1000]], dtype=torch.float32)
    
    ious = box_iou(proposals, gt_boxes)
    max_ious, max_idx = ious.max(dim=1)
    
    pos_mask = max_ious >= 0.5
    neg_mask = max_ious < 0.1
    
    pos_indices = torch.where(pos_mask)[0][:16]
    neg_indices = torch.where(neg_mask)[0][:48]
    sampled_indices = torch.cat([pos_indices, neg_indices])
    
    if len(sampled_indices) == 0:
        return 0.0
        
    sampled_proposals = proposals[sampled_indices]
    
    rois = torch.zeros((len(sampled_proposals), 5))
    rois[:, 0] = 0 
    rois[:, 1:] = sampled_proposals
    
    labels = torch.zeros(len(sampled_indices), dtype=torch.long)
    bbox_targets = torch.zeros((len(sampled_indices), 4))
    
    for i, idx in enumerate(sampled_indices):
        if idx in pos_indices:
            matched_gt = max_idx[idx]
            labels[i] = gt_labels[matched_gt]
            
            pw = proposals[idx, 2] - proposals[idx, 0] + 1e-6
            ph = proposals[idx, 3] - proposals[idx, 1] + 1e-6
            pcx = proposals[idx, 0] + pw * 0.5
            pcy = proposals[idx, 1] + ph * 0.5
            
            gw = gt_boxes[matched_gt, 2] - gt_boxes[matched_gt, 0]
            gh = gt_boxes[matched_gt, 3] - gt_boxes[matched_gt, 1]
            gcx = gt_boxes[matched_gt, 0] + gw * 0.5
            gcy = gt_boxes[matched_gt, 1] + gh * 0.5
            
            bbox_targets[i] = torch.tensor([
                (gcx - pcx) / pw,
                (gcy - pcy) / ph,
                torch.log(gw / pw),
                torch.log(gh / ph)
            ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0) 
    
    class_logits, bbox_offsets = model(img_tensor, rois)
    loss = criterion(class_logits, bbox_offsets, labels, bbox_targets)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()