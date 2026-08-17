"""
DETR-style detection loss for the new detection queries: Hungarian matching
(classification + L1 box + GIoU cost) followed by the matched loss. Standard
DETR-family formulation (same lineage as D-FINE, already a detection baseline in
this project) -- new code because Mask2Former has no detection component to reuse,
unlike the segmentation loss which reuses Mask2Former's own criterion directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_area(boxes):
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def generalized_box_iou(boxes1, boxes2):
    """boxes1: [N, 4], boxes2: [M, 4], both xyxy. Returns [N, M] GIoU matrix."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    """Bipartite matching between predicted and ground-truth boxes for one image,
    cost = weighted sum of classification cost + L1 box distance + GIoU cost.
    Standard DETR formulation. No learnable parameters."""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, class_logits, boxes, target_classes, target_boxes):
        """class_logits: [Q, num_classes+1], boxes: [Q, 4] (cxcywh, normalized).
        target_classes: [T], target_boxes: [T, 4] (cxcywh, normalized).
        Returns (query_indices, target_indices) for one image."""
        if target_classes.numel() == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        probs = class_logits.softmax(-1)
        cost_class = -probs[:, target_classes]  # [Q, T]

        cost_bbox = torch.cdist(boxes, target_boxes, p=1)  # [Q, T]

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes))

        cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        query_idx, target_idx = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(query_idx, dtype=torch.long), torch.as_tensor(target_idx, dtype=torch.long)


class DetectionLoss(nn.Module):
    """Matches then computes CE (with a no-object class for unmatched queries) +
    L1 box + GIoU loss, DETR-style. num_classes excludes the no-object class."""

    def __init__(self, num_classes, matcher=None, class_weight=1.0, bbox_weight=5.0, giou_weight=2.0,
                 no_object_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        # Down-weight the no-object class in the CE loss -- otherwise, with far
        # more unmatched queries than real objects per image, the loss is
        # dominated by "predict background" and never pushes hard on real matches.
        # Standard DETR practice.
        class_weights = torch.ones(num_classes + 1)
        class_weights[-1] = no_object_weight
        self.register_buffer('class_weights', class_weights)

    def forward(self, class_logits, boxes, targets):
        """class_logits: [B, Q, num_classes+1], boxes: [B, Q, 4].
        targets: list of B dicts, each {'classes': [T], 'boxes': [T, 4]} (cxcywh, normalized)."""
        B, Q = class_logits.shape[:2]
        no_object_class = self.num_classes

        target_class_full = torch.full((B, Q), no_object_class, dtype=torch.long, device=class_logits.device)
        bbox_losses, giou_losses = [], []

        for b in range(B):
            q_idx, t_idx = self.matcher(class_logits[b], boxes[b], targets[b]['classes'], targets[b]['boxes'])
            if q_idx.numel() == 0:
                continue
            target_class_full[b, q_idx] = targets[b]['classes'][t_idx].to(class_logits.device)

            matched_boxes = boxes[b, q_idx]
            matched_targets = targets[b]['boxes'][t_idx].to(boxes.device)
            bbox_losses.append(F.l1_loss(matched_boxes, matched_targets, reduction='sum'))
            giou = generalized_box_iou(box_cxcywh_to_xyxy(matched_boxes), box_cxcywh_to_xyxy(matched_targets))
            giou_losses.append((1 - giou.diagonal()).sum())

        class_loss = F.cross_entropy(class_logits.transpose(1, 2), target_class_full, weight=self.class_weights)

        n_matched = max(sum(t['classes'].numel() for t in targets), 1)
        bbox_loss = torch.stack(bbox_losses).sum() / n_matched if bbox_losses else class_logits.new_tensor(0.0)
        giou_loss = torch.stack(giou_losses).sum() / n_matched if giou_losses else class_logits.new_tensor(0.0)

        total = self.class_weight * class_loss + self.bbox_weight * bbox_loss + self.giou_weight * giou_loss
        return total, {'class_loss': class_loss.item(), 'bbox_loss': bbox_loss.item() if bbox_losses else 0.0,
                       'giou_loss': giou_loss.item() if giou_losses else 0.0}


if __name__ == '__main__':
    torch.manual_seed(0)
    B, Q, num_classes = 2, 100, 2

    class_logits = torch.randn(B, Q, num_classes + 1, requires_grad=True)
    boxes = torch.rand(B, Q, 4, requires_grad=True)

    targets = [
        {'classes': torch.tensor([0, 1]), 'boxes': torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.6, 0.2, 0.15]])},
        {'classes': torch.tensor([1]), 'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]])},
    ]

    loss_fn = DetectionLoss(num_classes=num_classes)
    loss, parts = loss_fn(class_logits, boxes, targets)
    print('detection loss:', loss.item(), 'parts:', parts)
    assert loss.item() == loss.item(), 'NaN loss'

    loss.backward()
    assert class_logits.grad is not None and class_logits.grad.abs().sum() > 0
    assert boxes.grad is not None and boxes.grad.abs().sum() > 0
    print('OK: Hungarian matcher + DETR-style detection loss verified (matching, gradients both check out).')

    # Sanity check: a query whose box is IDENTICAL to a target should be preferentially matched.
    matcher = HungarianMatcher()
    perfect_logits = torch.zeros(5, num_classes + 1)
    perfect_logits[2, 0] = 10.0  # query 2 confidently predicts class 0
    perfect_boxes = torch.rand(5, 4)
    perfect_boxes[2] = torch.tensor([0.3, 0.3, 0.1, 0.1])  # exact match to the target below
    q_idx, t_idx = matcher(perfect_logits, perfect_boxes, torch.tensor([0]), torch.tensor([[0.3, 0.3, 0.1, 0.1]]))
    print('matched query index for the exact-match case:', q_idx.tolist(), '(expected [2])')
    assert q_idx.tolist() == [2], 'Hungarian matcher failed to prefer the obviously correct match'
