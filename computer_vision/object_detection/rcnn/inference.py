import torch
import cv2
from torchvision import transforms
from torchvision.ops import nms
from model import RCNN

def inference(model_path, image_path, num_classes, conf_threshold=0.5, iou_threshold=0.5):
    model = RCNN(num_classes=num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    crops = []
    valid_proposals = []

    for x, y, w, h in rects:
        xmin, ymin, xmax, ymax = x, y, x + box_w, y + box_h
        crop = img_rgb[ymin:ymax, xmin:xmax]
        if crop.size == 0: continue
        crops.append(transform(crop))
        valid_proposals.append([xmin, ymin, xmax, ymax])

    proposals_tensor = torch.tensor(valid_proposals, dtype=torch.float32).to(device)
    inputs = torch.stack(crops).to(device)

    with torch.no_grad():
        class_logits, bbox_offsets = model(inputs)
        probs = torch.softmax(class_logits, dim=1)
        scores, labels = probs.max(dim=1)

    keep_mask = (labels > 0) & (scores > conf_threshold)
    
    if keep_mask.sum() == 0:
        print("No objects detected.")
        return
    
    final_proposals = proposals_tensor[keep_mask]
    final_scores = scores[keep_mask]
    final_labels = labels[keep_mask]
    final_offsets = bbox_offsets[keep_mask]

    p_w = final_proposals[:, 2] - final_proposals[:, 0]
    p_h = final_proposals[:, 3] - final_proposals[:, 1]
    p_cx = final_proposals[:, 0] + p_w * 0.5
    p_cy = final_proposals[:, 1] + p_h * 0.5
    
    refined_cx = final_offsets[:, 0] * p_w + p_cx
    refined_cy = final_offsets[:, 1] * p_h + p_cy
    refined_w = torch.exp(final_offsets[:, 2]) * p_w
    refined_h = torch.exp(final_offsets[:, 3]) * p_h
    
    refined_boxes = torch.zeros_like(final_proposals)
    refined_boxes[:, 0] = refined_cx - refined_w * 0.5
    refined_boxes[:, 1] = refined_cy - refined_h * 0.5
    refined_boxes[:, 2] = refined_cx + refined_w * 0.5
    refined_boxes[:, 3] = refined_cy + refined_h * 0.5

    keep_indices = nms(refined_boxes, final_scores, iou_threshold)
    for idx in keep_indices:
        box = refined_boxes[idx].int().cpu().numpy()
        score = final_scores[idx].item()
        label = final_labels[idx].item()
        print(f"Detected Class {label} at [{box[0]}, {box[1]}, {box[2]}, {box[3]}] with score: {score:.4f}")