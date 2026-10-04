"""Box prediction on top of Mask2Former's own queries, plus the two loss terms that make it
trainable without a second Hungarian matcher: query_box_loss (rides on the matched indices
Mask2Former's OWN class+mask matcher already computes -- see query_box_joint_model.py for how
those indices get extracted) and box_mask_consistency_loss (box vs. the box implied by the
model's own predicted mask, gradient-detached on the mask side -- see its docstring for why).

Deliberately decouples "who does this query belong to" (Mask2Former's existing, proven class+mask
matching, unchanged) from "what box should it predict" (a new regression head riding on that
already-decided assignment) -- the DETR-style failure mode this project hit repeatedly
(novelty_experiments/defectformer/, v1-v4) was trying to learn matching and box quality jointly
from a fresh, untrained matcher. This never has to.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# generalized_box_iou/box_cxcywh_to_xyxy are already correct and tested in defectformer/
# detection_loss.py (a sibling directory) -- reused directly rather than duplicated.
sys.path.insert(0, str(Path(__file__).parent.parent / "defectformer"))
from detection_loss import box_cxcywh_to_xyxy, generalized_box_iou


class QueryBoxHead(nn.Module):
    """Standard DETR-style 3-layer MLP box head, applied to the same per-query representation
    class_predictor/mask_predictor already consume. Outputs normalized cxcywh in [0,1], matching
    this project's box convention (load_yolo_boxes returns normalized cxcywh directly)."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, query_hidden_states):
        """query_hidden_states: [B,Q,hidden_dim] -> [B,Q,4] normalized cxcywh in [0,1]."""
        return self.mlp(query_hidden_states).sigmoid()


def query_box_loss(pred_boxes, gt_boxes_list, matched_indices, l1_weight=5.0, giou_weight=2.0):
    """pred_boxes: [B,Q,4] normalized cxcywh. gt_boxes_list: length-B list of [Ni,4] normalized
    cxcywh (Ni is however many GT instances that image's matcher call saw -- in this project's
    current Mask2Former convention that's always exactly 1, a single merged instance per image,
    but this doesn't hardcode that). matched_indices: length-B list of (index_i, index_j) exactly
    as returned by Mask2FormerHungarianMatcher -- index_i are matched query indices, index_j are
    the corresponding GT indices, same order. Loss weights (5.0/2.0) match DETR's own convention.
    """
    l1_total = pred_boxes.new_tensor(0.0)
    giou_total = pred_boxes.new_tensor(0.0)
    n_matched = 0

    for b, (index_i, index_j) in enumerate(matched_indices):
        if index_i.numel() == 0:
            continue
        matched_pred = pred_boxes[b, index_i]
        matched_gt = gt_boxes_list[b][index_j]
        l1_total = l1_total + torch.nn.functional.l1_loss(matched_pred, matched_gt, reduction="sum")
        giou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(matched_pred), box_cxcywh_to_xyxy(matched_gt)))
        giou_total = giou_total + (1.0 - giou).sum()
        n_matched += index_i.numel()

    if n_matched == 0:
        return pred_boxes.new_tensor(0.0), {"l1": pred_boxes.new_tensor(0.0), "giou": pred_boxes.new_tensor(0.0)}

    l1_mean = l1_total / n_matched
    giou_mean = giou_total / n_matched
    total = l1_weight * l1_mean + giou_weight * giou_mean
    return total, {"l1": l1_mean.detach(), "giou": giou_mean.detach()}


def bbox_from_mask(mask_probs, threshold=0.5):
    """mask_probs: [*, H, W] sigmoid probabilities (already, not logits). Returns (boxes [*,4]
    normalized cxcywh, valid [*] bool -- False where the thresholded mask has zero positive
    pixels, meaning no box could be derived). Uses a hard threshold -- not differentiable through
    the mask, which is deliberate: this box is meant as a fixed pseudo-target each forward pass
    (see box_mask_consistency_loss), not a path for the box loss to update the mask itself
    (the mask already has its own proper mask/dice loss elsewhere)."""
    orig_shape = mask_probs.shape[:-2]
    H, W = mask_probs.shape[-2:]
    flat = mask_probs.reshape(-1, H, W)
    binary = flat > threshold

    boxes = flat.new_zeros(flat.shape[0], 4)
    valid = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
    for i in range(flat.shape[0]):
        ys, xs = torch.where(binary[i])
        if ys.numel() == 0:
            continue
        x1, x2 = xs.min().float(), xs.max().float()
        y1, y2 = ys.min().float(), ys.max().float()
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1 + 1) / W, (y2 - y1 + 1) / H
        boxes[i] = torch.tensor([cx, cy, w, h], device=flat.device)
        valid[i] = True

    return boxes.reshape(*orig_shape, 4), valid.reshape(*orig_shape)


def box_mask_consistency_loss(pred_boxes, matched_mask_probs, matched_indices):
    """pred_boxes: [B,Q,4] normalized cxcywh (the SAME query box head output query_box_loss
    uses). matched_mask_probs: [B,Q,H,W] sigmoid mask probabilities for those same queries.
    matched_indices: same format as query_box_loss. For each matched query, derives a box from
    its OWN predicted mask (detached -- see bbox_from_mask) and pulls the predicted box toward
    it via L1. This is a consistency check between two of the model's own outputs, not a second
    ground-truth supervision signal."""
    total = pred_boxes.new_tensor(0.0)
    n_valid = 0

    for b, (index_i, index_j) in enumerate(matched_indices):
        if index_i.numel() == 0:
            continue
        matched_pred_boxes = pred_boxes[b, index_i]
        matched_mask = matched_mask_probs[b, index_i].detach()  # detached: this loss trains the box, not the mask
        derived_boxes, valid = bbox_from_mask(matched_mask)
        if not valid.any():
            continue
        total = total + torch.nn.functional.l1_loss(matched_pred_boxes[valid], derived_boxes[valid], reduction="sum")
        n_valid += int(valid.sum().item())

    if n_valid == 0:
        return pred_boxes.new_tensor(0.0)
    return total / n_valid


if __name__ == "__main__":
    torch.manual_seed(0)

    print("QueryBoxHead shape + gradient check")
    head = QueryBoxHead(hidden_dim=256)
    queries = torch.randn(2, 100, 256, requires_grad=True)
    boxes = head(queries)
    assert boxes.shape == (2, 100, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all(), "sigmoid output must be in [0,1]"
    boxes.sum().backward()
    assert queries.grad is not None and queries.grad.abs().sum() > 0

    print("\nbbox_from_mask: exact rectangle recovery")
    mask = torch.zeros(1, 20, 20)
    mask[0, 4:9, 5:12] = 1.0  # rows 4-8, cols 5-11 -> x in [5,11], y in [4,8]
    boxes_out, valid = bbox_from_mask(mask)
    assert valid[0].item()
    cx, cy, w, h = boxes_out[0].tolist()
    x1, x2 = (cx - w / 2) * 20, (cx + w / 2) * 20
    y1, y2 = (cy - h / 2) * 20, (cy + h / 2) * 20
    print(f"recovered box (pixels): x=[{x1:.1f},{x2:.1f}] y=[{y1:.1f},{y2:.1f}], expected x=[5,12] y=[4,9]")
    assert abs(x1 - 5) < 0.6 and abs(x2 - 12) < 0.6 and abs(y1 - 4) < 0.6 and abs(y2 - 9) < 0.6

    print("\nbbox_from_mask: empty mask handled without crashing")
    empty_mask = torch.zeros(1, 20, 20)
    _, valid_empty = bbox_from_mask(empty_mask)
    assert not valid_empty[0].item()
    print("OK: empty mask correctly flagged invalid, no NaN/crash.")

    print("\nquery_box_loss: perfect prediction on the matched query -> near-zero loss")
    pred_boxes = torch.zeros(1, 5, 4)
    gt_boxes_list = [torch.tensor([[0.5, 0.5, 0.2, 0.3]])]
    pred_boxes[0, 2] = torch.tensor([0.5, 0.5, 0.2, 0.3])  # query 2 is the matched one, predicts exactly right
    matched_indices = [(torch.tensor([2]), torch.tensor([0]))]
    loss, parts = query_box_loss(pred_boxes, gt_boxes_list, matched_indices)
    print(f"loss={loss.item():.6f}, parts={parts}")
    assert loss.item() < 1e-4, "a perfect matched prediction should give ~0 loss"

    print("\nquery_box_loss: wrong prediction on the matched query -> real, nonzero loss")
    pred_boxes_wrong = torch.zeros(1, 5, 4)
    pred_boxes_wrong[0, 2] = torch.tensor([0.1, 0.1, 0.05, 0.05])
    loss_wrong, _ = query_box_loss(pred_boxes_wrong, gt_boxes_list, matched_indices)
    print(f"loss={loss_wrong.item():.4f}")
    assert loss_wrong.item() > 1.0, "a badly wrong box should score clearly worse"

    print("\nquery_box_loss: unmatched queries (not index 2) must not affect the loss at all")
    pred_boxes_extra_junk = pred_boxes.clone()
    pred_boxes_extra_junk[0, 0] = torch.tensor([0.9, 0.9, 0.9, 0.9])  # garbage in an unmatched slot
    loss_junk, _ = query_box_loss(pred_boxes_extra_junk, gt_boxes_list, matched_indices)
    assert abs(loss_junk.item() - loss.item()) < 1e-6, "garbage in an unmatched query must not leak into the loss"
    print("OK: only the matched query's prediction affects the loss.")

    print("\nbox_mask_consistency_loss: box exactly matching its own query's mask -> ~0 loss")
    mask_probs = torch.zeros(1, 5, 20, 20)
    mask_probs[0, 2, 4:9, 5:12] = 1.0
    pred_boxes_consistent = torch.zeros(1, 5, 4)
    derived, _ = bbox_from_mask(mask_probs[0, 2:3])
    pred_boxes_consistent[0, 2] = derived[0]
    cons_loss = box_mask_consistency_loss(pred_boxes_consistent, mask_probs, matched_indices)
    print(f"consistency loss (matching box/mask): {cons_loss.item():.6f}")
    assert cons_loss.item() < 1e-4

    print("\nbox_mask_consistency_loss: mismatched box/mask -> real, nonzero loss")
    pred_boxes_mismatch = torch.zeros(1, 5, 4)
    pred_boxes_mismatch[0, 2] = torch.tensor([0.1, 0.1, 0.05, 0.05])
    cons_loss_bad = box_mask_consistency_loss(pred_boxes_mismatch, mask_probs, matched_indices)
    print(f"consistency loss (mismatched): {cons_loss_bad.item():.4f}")
    assert cons_loss_bad.item() > 0.3

    print("\nGradient check: consistency loss must update the BOX prediction, not the mask")
    queries2 = torch.randn(1, 5, 256, requires_grad=True)
    mask_logits = torch.randn(1, 5, 20, 20, requires_grad=True)
    boxes2 = head(queries2)
    cons_loss2 = box_mask_consistency_loss(boxes2, mask_logits.sigmoid(), matched_indices)
    cons_loss2.backward()
    assert queries2.grad is not None and queries2.grad.abs().sum() > 0, "box side must receive gradient"
    assert mask_logits.grad is None or mask_logits.grad.abs().sum() == 0, \
        "mask side must NOT receive gradient from this loss -- it's detached by design (own mask/dice loss owns that)"
    print("OK: gradient flows into the box prediction only, mask side correctly detached.")

    print("\nAll query_box_head self-tests passed.")
