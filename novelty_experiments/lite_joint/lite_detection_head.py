"""Decoupled anchor-free detection head (YOLOX/YOLOv8-style: separate classification and
regression conv towers, DFL-based box regression) for the lightweight joint model. Head
weights are SHARED across all 4 pyramid levels (one tower applied independently at each scale,
not 4 separate towers) -- the standard RetinaNet/FCOS convention, and the deliberate choice that
keeps this "lightweight" on parameter count given the shared Swin backbone is already the
dominant cost.
"""
import torch
import torch.nn as nn
from torchvision.ops import nms

from swin_fpn_neck import ConvBNAct
from dfl_loss import DFLIntegral
from tal_assigner import make_anchor_points


class DecoupledDetectHead(nn.Module):
    def __init__(self, in_ch=128, num_classes=1, reg_max=16, tower_ch=64, prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_tower = nn.Sequential(ConvBNAct(in_ch, tower_ch, k=3), ConvBNAct(tower_ch, tower_ch, k=3))
        self.reg_tower = nn.Sequential(ConvBNAct(in_ch, tower_ch, k=3), ConvBNAct(tower_ch, tower_ch, k=3))
        self.cls_pred = nn.Conv2d(tower_ch, num_classes, 1)
        self.reg_pred = nn.Conv2d(tower_ch, 4 * reg_max, 1)

        # RetinaNet/focal-loss-style prior: start the classifier predicting "background" with
        # high confidence for every anchor, not ~0.5. Without this, a dense anchor-free head's
        # early training is dominated by gradient from thousands of confidently-wrong-positive
        # anchors, which is exactly the kind of early instability this project has repeatedly
        # been burned by (see novelty_experiments/v1-v4's detection failures) -- cheap to add,
        # well-documented fix, worth having from the start rather than discovering the need for
        # it after another failed run.
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value.item())

        self.integral = DFLIntegral(reg_max)

    def forward(self, features_per_level):
        """features_per_level: [P2,P3,P4,P5], each [B,in_ch,H,W]. Returns cls_logits [B,N,C]
        and reg_logits [B,N,4,reg_max] concatenated across all levels (N = sum of H*W)."""
        cls_outs, reg_outs = [], []
        for feat in features_per_level:
            B, _, H, W = feat.shape
            cls_outs.append(self.cls_pred(self.cls_tower(feat)).permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes))
            reg_outs.append(self.reg_pred(self.reg_tower(feat)).permute(0, 2, 3, 1).reshape(B, H * W, 4 * self.reg_max))
        cls_logits = torch.cat(cls_outs, dim=1)
        reg_logits = torch.cat(reg_outs, dim=1).view(cls_logits.shape[0], -1, 4, self.reg_max)
        return cls_logits, reg_logits

    def decode(self, cls_logits, reg_logits, anchor_points, anchor_strides):
        """-> scores [B,N,C] (sigmoid probs), boxes_xyxy [B,N,4] (pixel space)."""
        scores = cls_logits.sigmoid()
        ltrb = self.integral(reg_logits) * anchor_strides.view(1, -1, 1)
        ax, ay = anchor_points[:, 0].view(1, -1), anchor_points[:, 1].view(1, -1)
        boxes = torch.stack([ax - ltrb[..., 0], ay - ltrb[..., 1], ax + ltrb[..., 2], ay + ltrb[..., 3]], dim=-1)
        return scores, boxes

    @staticmethod
    def postprocess(scores, boxes, score_thresh=0.05, iou_thresh=0.6, max_dets=300):
        """Single-class-friendly NMS per image. scores: [B,N,C], boxes: [B,N,4]. Returns a list
        of (boxes_xyxy [K,4], scores [K], labels [K]) per image, K possibly 0."""
        results = []
        B, N, C = scores.shape
        for b in range(B):
            all_boxes, all_scores, all_labels = [], [], []
            for c in range(C):
                keep_score = scores[b, :, c] > score_thresh
                if keep_score.sum() == 0:
                    continue
                cand_boxes = boxes[b][keep_score]
                cand_scores = scores[b, :, c][keep_score]
                keep = nms(cand_boxes, cand_scores, iou_thresh)[:max_dets]
                all_boxes.append(cand_boxes[keep])
                all_scores.append(cand_scores[keep])
                all_labels.append(torch.full((keep.shape[0],), c, dtype=torch.long, device=scores.device))
            if all_boxes:
                results.append((torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)))
            else:
                results.append((boxes.new_zeros(0, 4), scores.new_zeros(0), torch.zeros(0, dtype=torch.long, device=scores.device)))
        return results


if __name__ == "__main__":
    from swin_fpn_neck import SwinPAFPNNeck

    torch.manual_seed(0)
    print("Chaining a fresh SwinPAFPNNeck (random weights, but the same class already verified")
    print("against the real checkpoint in swin_fpn_neck.py) on real-shaped stage tensors, so this")
    print("head is tested against genuine neck output, not hand-typed placeholder shapes.")
    stage_shapes = [(1, 96, 160, 160), (1, 192, 80, 80), (1, 384, 40, 40), (1, 768, 20, 20)]
    stage_features = [torch.randn(*s) for s in stage_shapes]
    neck = SwinPAFPNNeck(in_channels=(96, 192, 384, 768), neck_ch=128, n=1)
    neck.train()
    neck_out = neck(stage_features)

    head = DecoupledDetectHead(in_ch=128, num_classes=1, reg_max=16, tower_ch=64)
    head.train()
    cls_logits, reg_logits = head(neck_out)
    N_expected = 160 * 160 + 80 * 80 + 40 * 40 + 20 * 20
    print(f"cls_logits: {tuple(cls_logits.shape)}, reg_logits: {tuple(reg_logits.shape)}, expected N={N_expected}")
    assert cls_logits.shape == (1, N_expected, 1)
    assert reg_logits.shape == (1, N_expected, 4, 16)

    print("\nPrior-bias check: at init, sigmoid(cls_logits) should be near 0.01 everywhere (background prior),")
    print("not ~0.5 -- confirming the stability fix above actually took effect.")
    mean_prob = cls_logits.sigmoid().mean().item()
    print(f"mean predicted foreground probability at init: {mean_prob:.4f}")
    assert 0.005 < mean_prob < 0.02, f"expected ~0.01 from the prior-bias init, got {mean_prob}"

    print("\nDecode check: anchor points + regression -> plausible pixel-space boxes")
    anchor_points, anchor_strides = make_anchor_points([s[-2:] for s in stage_shapes], [4, 8, 16, 32])
    assert anchor_points.shape[0] == N_expected
    scores, boxes = head.decode(cls_logits, reg_logits, anchor_points, anchor_strides)
    print("scores range:", scores.min().item(), scores.max().item())
    print("boxes sample (first 3):", boxes[0, :3].tolist())
    assert (scores >= 0).all() and (scores <= 1).all()
    assert (boxes[..., 2] >= boxes[..., 0]).all() and (boxes[..., 3] >= boxes[..., 1]).all(), \
        "decoded x2 must be >= x1 and y2 >= y1 -- DFL distances are non-negative by construction (clamped in encode_dfl_targets), so this should hold automatically"

    print("\nPostprocess (NMS) check: runs without error and returns a valid, possibly-empty result per image")
    results = head.postprocess(scores, boxes, score_thresh=0.5, iou_thresh=0.6)
    kept_boxes, kept_scores, kept_labels = results[0]
    print(f"kept {kept_boxes.shape[0]} boxes after score>0.5 + NMS (at random init, expect very few or 0 --")
    print("that's the prior-bias fix working as intended, not a bug)")

    print("\nGradient coverage check: backward through a dummy loss on cls+reg outputs")
    loss = cls_logits.sum() + reg_logits.sum()
    loss.backward()
    n_total = sum(1 for p in head.parameters() if p.requires_grad)
    n_grad = sum(1 for p in head.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Head: {n_grad}/{n_total} params with nonzero grad")
    assert n_grad == n_total, "every head parameter must receive gradient (shared across all 4 levels, so all 4 levels' loss must reach every param)"

    print("\nOK: DecoupledDetectHead verified -- correct shapes, background-prior init confirmed numerically,")
    print("valid decode geometry, NMS runs cleanly, full gradient coverage through the shared head.")
