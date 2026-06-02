from torch import nn

class FasterRCNNLoss(nn.Module):
    def __init__(self):
        super(FasterRCNNLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, model_outputs, targets):
        rpn_cls_logits, rpn_bbox_offsets, final_cls_logits, final_bbox_offsets = model_outputs
        
        rpn_target_labels = targets['rpn_labels']      
        rpn_target_offsets = targets['rpn_offsets']    
        final_target_labels = targets['final_labels']  
        final_target_offsets = targets['final_offsets']

        loss_rpn_cls = self.cls_loss(rpn_cls_logits, rpn_target_labels)
        loss_rpn_reg = self.reg_loss(rpn_bbox_offsets, rpn_target_offsets)
        
        loss_final_cls = self.cls_loss(final_cls_logits, final_target_labels)
        loss_final_reg = self.reg_loss(final_bbox_offsets, final_target_offsets)
        
        total_loss = loss_rpn_cls + loss_rpn_reg + loss_final_cls + loss_final_reg
        return total_loss