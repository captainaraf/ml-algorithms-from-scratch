import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import RCNN_Dataset
from model import RCNN
from tqdm.auto import tqdm

def train_epoch(model, dataloader, criterion_cls, optimizer, criterion_bbox, device):
    model.train()
    total_loss = 0.0

    for images, labels, bboxes in tqdm(dataloader):
        images = images.view(-1, 3, 224, 224).to(device)
        labels = labels.view(-1).to(device)
        bbox_targets = bbox_targets.view(-1, 4).to(device)

        if images.size(0) == 0:
            continue

        optimizer.zero_grad()
        
        class_logits, bbox_offsets = model(images)
        
        loss_cls = criterion_cls(class_logits, labels)
        pos_mask = labels > 0
        if pos_mask.sum() > 0:
            loss_bbox = criterion_bbox(bbox_offsets[pos_mask], bbox_targets[pos_mask])
        else:
            loss_bbox = torch.tensor(0.0).to(device)

        loss = loss_cls + 2.0 * loss_bbox
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)