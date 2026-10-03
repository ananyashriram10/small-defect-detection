"""YOLO-SM training loss with a compact SimOTA implementation."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-6)


def ciou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou = box_iou(boxes1, boxes2).diag()
    c1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
    c2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    center_dist = ((c1 - c2) ** 2).sum(dim=1)
    enclose_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclose_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    diagonal = ((enclose_rb - enclose_lt) ** 2).sum(dim=1).clamp(min=1e-6)
    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6)
    v = (4 / 3.1415926535**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    alpha = v / (1 - iou + v).clamp(min=1e-6)
    return 1 - (iou - center_dist / diagonal - alpha * v)


def simota_assign(
    pred_boxes: Tensor,
    cls_logits: Tensor,
    obj_logits: Tensor,
    points: Tensor,
    strides: Tensor,
    gt_boxes: Tensor,
    topk: int = 10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return foreground mask, matched GT indices, and matched IoUs."""
    n = pred_boxes.shape[0]
    if gt_boxes.numel() == 0:
        return pred_boxes.new_zeros(n, dtype=torch.bool), pred_boxes.new_zeros(n, dtype=torch.long), pred_boxes.new_zeros(n)

    centers = points
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    half = strides[:, None] * 2.5
    in_box = (
        (centers[:, None, 0] >= gt_boxes[None, :, 0])
        & (centers[:, None, 1] >= gt_boxes[None, :, 1])
        & (centers[:, None, 0] <= gt_boxes[None, :, 2])
        & (centers[:, None, 1] <= gt_boxes[None, :, 3])
    )
    in_center = (
        (centers[:, None, 0] >= gt_centers[None, :, 0] - half)
        & (centers[:, None, 1] >= gt_centers[None, :, 1] - half)
        & (centers[:, None, 0] <= gt_centers[None, :, 0] + half)
        & (centers[:, None, 1] <= gt_centers[None, :, 1] + half)
    )
    candidate = in_box | in_center
    ious = box_iou(pred_boxes, gt_boxes)
    cls_prob = cls_logits.sigmoid().squeeze(-1)
    obj_prob = obj_logits.sigmoid().squeeze(-1)
    joint_prob = (cls_prob * obj_prob).clamp(min=1e-6, max=1 - 1e-6)
    cls_cost = -torch.log(joint_prob)[:, None].expand(-1, gt_boxes.shape[0])
    cost = cls_cost + 3.0 * (1.0 - ious)
    cost = cost + (~candidate).float() * 1e5

    selected = []
    for gt_idx in range(gt_boxes.shape[0]):
        valid_ious = ious[:, gt_idx].masked_fill(~candidate[:, gt_idx], 0)
        k = max(1, min(topk, valid_ious.numel(), int(valid_ious.topk(min(topk, valid_ious.numel())).values.sum().item())))
        candidates = torch.where(candidate[:, gt_idx])[0]
        if candidates.numel() == 0:
            candidates = torch.tensor([int(torch.argmin(cost[:, gt_idx]))], device=pred_boxes.device)
        k = min(k, candidates.numel())
        local_cost = cost[candidates, gt_idx]
        chosen = candidates[torch.topk(local_cost, k=k, largest=False).indices]
        selected.extend((int(idx), gt_idx) for idx in chosen.tolist())

    fg = pred_boxes.new_zeros(n, dtype=torch.bool)
    matched_gt = pred_boxes.new_zeros(n, dtype=torch.long)
    matched_iou = pred_boxes.new_zeros(n)
    best_cost = pred_boxes.new_full((n,), float("inf"))
    for prior_idx, gt_idx in selected:
        if cost[prior_idx, gt_idx] < best_cost[prior_idx]:
            best_cost[prior_idx] = cost[prior_idx, gt_idx]
            fg[prior_idx] = True
            matched_gt[prior_idx] = gt_idx
            matched_iou[prior_idx] = ious[prior_idx, gt_idx]
    return fg, matched_gt, matched_iou


def yolosm_loss(outputs: dict[str, Tensor], gt_boxes_batch: list[Tensor], num_classes: int = 1) -> tuple[Tensor, dict[str, float]]:
    cls_logits = outputs["cls_logits"]
    obj_logits = outputs["obj_logits"]
    pred_boxes = outputs["boxes"]
    points = outputs["points"]
    strides = outputs["strides"]
    batch_size = cls_logits.shape[0]

    total_obj = cls_logits.new_tensor(0.0)
    total_cls = cls_logits.new_tensor(0.0)
    total_reg = cls_logits.new_tensor(0.0)
    total_fg = 0

    for batch_idx in range(batch_size):
        gt_boxes = gt_boxes_batch[batch_idx].to(cls_logits.device)
        fg, matched_gt, matched_iou = simota_assign(
            pred_boxes[batch_idx], cls_logits[batch_idx], obj_logits[batch_idx], points, strides, gt_boxes
        )
        obj_target = obj_logits[batch_idx].new_zeros(obj_logits.shape[1], 1)
        obj_target[fg] = 1.0
        total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits[batch_idx], obj_target, reduction="mean")

        if fg.any():
            pos_idx = torch.where(fg)[0]
            pos_iou = matched_iou[pos_idx].detach().clamp(0, 1)
            cls_target = pos_iou[:, None].expand(-1, num_classes)
            total_cls = total_cls + F.binary_cross_entropy_with_logits(cls_logits[batch_idx, pos_idx], cls_target, reduction="mean")
            total_reg = total_reg + ciou_loss(pred_boxes[batch_idx, pos_idx], gt_boxes[matched_gt[pos_idx]]).mean()
            total_fg += int(pos_idx.numel())

    normalizer = max(batch_size, 1)
    loss = (total_obj / normalizer) + (total_cls / normalizer) + (total_reg / normalizer)
    return loss, {
        "loss": float(loss.detach()),
        "obj_loss": float((total_obj / normalizer).detach()),
        "cls_loss": float((total_cls / normalizer).detach()),
        "box_loss": float((total_reg / normalizer).detach()),
        "foreground_priors": float(total_fg / normalizer),
    }

