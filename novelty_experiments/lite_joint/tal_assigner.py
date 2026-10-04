"""Task-aligned label assignment for the lightweight anchor-free detection head (TOOD /
YOLOv8-style TAL) -- replaces DefectFormer's Hungarian bipartite matching, which is the
component root-caused (this session, across v1-v4) to detection's repeated failure. TAL
assigns anchors to GT boxes deterministically from a live alignment metric (classification
score x IoU) rather than solving a combinatorial optimum every step, which is what makes it
converge fast and stably even from random init -- the actual reason it's the right fix, not
just a swap for its own sake.

No learnable parameters -- same framing as HungarianMatcher in detection_loss.py.
"""
import torch
import torch.nn as nn


def make_anchor_points(feature_shapes, strides, device="cpu"):
    """feature_shapes: list of (H, W) per pyramid level. strides: matching list of ints.
    Returns (anchor_points [N,2] xy pixel-space centers, anchor_strides [N]) concatenated
    across all levels, in the same order levels are processed elsewhere."""
    points, level_strides = [], []
    for (h, w), stride in zip(feature_shapes, strides):
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        points.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1))
        level_strides.append(torch.full((h * w,), stride, device=device, dtype=torch.float32))
    return torch.cat(points, dim=0), torch.cat(level_strides, dim=0)


def box_iou_xyxy(boxes1, boxes2):
    """[N,4] x [M,4] xyxy -> [N,M] plain IoU (not GIoU -- TAL's own alignment metric uses
    plain IoU by convention, distinct from the GIoU regression loss used elsewhere)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, anchor_points, pred_scores, pred_boxes, gt_boxes_list, gt_classes_list, num_classes):
        """anchor_points: [N,2] xy. pred_scores: [B,N,C] sigmoid probs. pred_boxes: [B,N,4] xyxy,
        same pixel space as anchor_points. gt_boxes_list/gt_classes_list: length-B lists of
        [Mi,4] xyxy / [Mi] (Mi may be 0). Returns target_scores [B,N,C], target_boxes [B,N,4],
        fg_mask [B,N] bool."""
        B, N, C = pred_scores.shape
        device = pred_scores.device
        target_scores = torch.zeros(B, N, C, device=device)
        target_boxes = torch.zeros(B, N, 4, device=device)
        fg_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            gt_boxes = gt_boxes_list[b]
            gt_classes = gt_classes_list[b]
            M = gt_boxes.shape[0]
            if M == 0:
                continue

            cx, cy = anchor_points[:, 0], anchor_points[:, 1]
            inside = (
                (cx[:, None] > gt_boxes[None, :, 0]) & (cx[:, None] < gt_boxes[None, :, 2]) &
                (cy[:, None] > gt_boxes[None, :, 1]) & (cy[:, None] < gt_boxes[None, :, 3])
            )  # [N, M]

            iou = box_iou_xyxy(pred_boxes[b], gt_boxes)  # [N, M]
            cls_score = pred_scores[b][:, gt_classes]  # [N, M], gathers each GT's own class column
            align_metric = (cls_score.clamp(min=0) ** self.alpha) * (iou.clamp(min=0) ** self.beta)
            align_metric = align_metric * inside  # zero out anchors whose center isn't in the box

            k = min(self.topk, N)
            topk_vals, topk_idx = align_metric.topk(k, dim=0)  # [k, M]
            topk_mask = topk_vals > self.eps  # drop picks that were only nonzero due to containment but metric==0
            candidate = torch.zeros(N, M, dtype=torch.bool, device=device)
            candidate.scatter_(0, topk_idx, topk_mask)

            # an anchor claimed by >1 GT keeps only the GT it has highest IoU with (TOOD convention)
            claim_count = candidate.sum(dim=1)
            contested = claim_count > 1
            if contested.any():
                best_gt = iou[contested].argmax(dim=1)
                candidate[contested] = False
                candidate[contested, best_gt] = True

            anchor_is_fg = candidate.any(dim=1)  # [N]
            assigned_gt = candidate.float().argmax(dim=1)  # [N], meaningless where anchor_is_fg is False

            fg_mask[b] = anchor_is_fg
            target_boxes[b][anchor_is_fg] = gt_boxes[assigned_gt[anchor_is_fg]]

            # per-GT normalization: rescale each GT's winning alignment metrics so its best
            # anchor's soft target equals that GT's own best achieved IoU (standard TAL scaling,
            # keeps the classification target meaningful in [0,1] rather than in raw metric units)
            max_align_per_gt = (align_metric * candidate).amax(dim=0)  # [M]
            max_iou_per_gt = (iou * candidate).amax(dim=0)  # [M]
            scale = (max_iou_per_gt / max_align_per_gt.clamp(min=self.eps)).clamp(max=1.0 / self.eps)
            soft_target = align_metric[anchor_is_fg, assigned_gt[anchor_is_fg]] * scale[assigned_gt[anchor_is_fg]]
            target_scores[b][anchor_is_fg, gt_classes[assigned_gt[anchor_is_fg]]] = soft_target.clamp(0, 1)

        return target_scores, target_boxes, fg_mask


if __name__ == "__main__":
    torch.manual_seed(0)
    assigner = TaskAlignedAssigner(topk=1, alpha=1.0, beta=6.0)

    print("Case 1: one anchor exactly at GT center with a perfect prediction, others far away")
    anchor_points = torch.tensor([[50.0, 50.0], [500.0, 500.0], [10.0, 500.0]])
    gt_boxes = [torch.tensor([[20.0, 20.0, 80.0, 80.0]])]
    gt_classes = [torch.tensor([0])]
    pred_scores = torch.zeros(1, 3, 1)
    pred_scores[0, 0, 0] = 0.95  # the center anchor predicts high confidence
    pred_scores[0, 1, 0] = 0.9   # a far anchor ALSO predicts high confidence, but its box IoU will be 0
    pred_boxes = torch.zeros(1, 3, 4)
    pred_boxes[0, 0] = torch.tensor([20.0, 20.0, 80.0, 80.0])  # anchor 0: exact GT box, IoU=1
    pred_boxes[0, 1] = torch.tensor([480.0, 480.0, 520.0, 520.0])  # anchor 1: nowhere near GT
    pred_boxes[0, 2] = torch.tensor([0.0, 480.0, 20.0, 520.0])
    ts, tb, fg = assigner(anchor_points, pred_scores, pred_boxes, gt_boxes, gt_classes, num_classes=1)
    print("fg_mask:", fg.tolist())
    assert fg[0, 0].item() is True or fg[0, 0].item() == 1, "the anchor at the GT center with a perfect box must be selected"
    assert not fg[0, 1] and not fg[0, 2], "anchors outside the GT box must never be selected regardless of confidence"
    assert torch.allclose(tb[0, 0], gt_boxes[0][0]), "assigned target box must be the real GT box"
    print("OK: center/high-IoU anchor selected, high-confidence-but-wrong-location anchor correctly rejected.\n")

    print("Case 2: two non-overlapping GTs, each anchor must bind to its own nearby GT, not the other's")
    anchor_points2 = torch.tensor([[50.0, 50.0], [550.0, 550.0]])
    gt_boxes2 = [torch.tensor([[20.0, 20.0, 80.0, 80.0], [520.0, 520.0, 580.0, 580.0]])]
    gt_classes2 = [torch.tensor([0, 0])]
    pred_scores2 = torch.full((1, 2, 1), 0.9)
    pred_boxes2 = torch.zeros(1, 2, 4)
    pred_boxes2[0, 0] = torch.tensor([20.0, 20.0, 80.0, 80.0])
    pred_boxes2[0, 1] = torch.tensor([520.0, 520.0, 580.0, 580.0])
    ts2, tb2, fg2 = assigner(anchor_points2, pred_scores2, pred_boxes2, gt_boxes2, gt_classes2, num_classes=1)
    assert fg2.all(), "both anchors should be assigned, each to its own nearby GT"
    assert torch.allclose(tb2[0, 0], gt_boxes2[0][0]) and torch.allclose(tb2[0, 1], gt_boxes2[0][1]), \
        "each anchor must bind to the GT it's actually inside, not cross-assigned"
    print("OK: no cross-assignment between spatially separate GTs.\n")

    print("Case 3: an image with zero GT boxes must not crash and must produce an all-background result")
    ts3, tb3, fg3 = assigner(anchor_points2, pred_scores2, pred_boxes2, [torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.long)], num_classes=1)
    assert not fg3.any(), "no GT boxes means no anchor should ever be foreground"
    print("OK: zero-GT image handled without crashing, all anchors background.\n")

    print("All TaskAlignedAssigner self-tests passed.")
