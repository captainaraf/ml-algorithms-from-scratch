import cv2 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import box_iou

class RCNN_Dataset(Dataset):
    def __init__(self, image_paths, gt_boxes, gt_labels, num_classes, max_proposals=2000):
        self.image_paths = image_paths
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.num_classes = num_classes
        self.max_proposals = max_proposals
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def _get_selective_search_proposals(self, image):
        selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        selective_search.setBaseImage(image)
        selective_search.switchToSelectiveSearchFast()
        rectangles = selective_search.process()

        proposals = []
        for (x, y, w, h) in rectangles[:self.max_proposals]:
            proposals.append([x, y, x + w, y + h])
        
        return torch.tensor(proposals, dtype=torch.float32)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        gt_b = self.gt_boxes[idx]
        gt_l = self.gt_labels[idx]

        proposals = self._get_selective_search_proposals(img)
        if len(proposals) == 0:
            proposals = torch.tensor([[0, 0, w, h]], dtype=torch.float3)

        proposals[:, [0, 2]] = proposals[:, [0, 2]].clamp(0, w)
        proposals[:, [1, 3]] = proposals[:, [1, 3]].clamp(0, h)

        ious = box_iou(proposals, gt_b)
        max_ious, max_idx = ious.max(dim=1)

        pos_mask = max_ious >= 0.5
        neg_mask = max_ious < 0.1

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        sampled_pos = pos_indices[torch.randperm(len(pos_indices))[:32]]
        needed_neg = 128 - len(sampled_pos)
        sampled_neg = neg_indices[torch.randperm(len(neg_indices))[:needed_neg]]

        sampled_indices = torch.cat([sampled_pos, sampled_neg])

        crops = []
        labels = []
        bbox_targets = []

        for i in sampled_indices:
            box = proposals[i].int().tolist()
            crop = img[box[1]:box[3], box[0]:box[2]]

            if crop.size == 0:
                continue

            crops.append(self.transform(crop))
            if i in sampled_pos:
                matched_gt_idx = max_idx[i]
                labels.append(gt_l[matched_gt_idx])

                p_w = proposals[i, 2] - proposals[i, 0] + 1e-6
                p_h = proposals[i, 3] - proposals[i, 1] + 1e-6
                p_cx = proposals[i, 0] + p_w * 0.5
                p_cy = proposals[i, 1] + p_h * 0.5

                g_w = gt_b[matched_gt_idx, 2] - gt_b[matched_gt_idx, 0]
                g_h = gt_b[matched_gt_idx, 3] - gt_b[matched_gt_idx, 1]
                g_cx = gt_b[matched_gt_idx, 0] + g_w * 0.5
                g_cy = gt_b[matched_gt_idx, 1] + g_h * 0.5

                tx = (g_cx - p_cx) / p_w
                ty = (g_cy - p_cy) / p_h
                tw = torch.log(g_w / p_w)
                th = torch.log(g_h / p_h)
                bbox_targets.append(torch.tensor([tx, ty, tw, th]))

            else:
                labels.append(torch.tensor(0))
                bbox_targets.append(torch.zeros(4))

        return torch.stack(crops), torch.stack(labels), torch.stack(bbox_targets)