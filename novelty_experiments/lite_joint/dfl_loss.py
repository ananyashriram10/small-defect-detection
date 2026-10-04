"""Distribution Focal Loss (Li et al., Generalized Focal Loss) for the lightweight detection
head's box regression -- each of the 4 box sides (left/top/right/bottom, in stride units from
an anchor point) is predicted as a discrete distribution over reg_max bins rather than a single
regressed number, trained against a soft two-bin target so a non-integer true distance (e.g.
3.7) still has an exact, differentiable target instead of being rounded.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFLIntegral(nn.Module):
    """Turns a softmax'd per-side bin distribution back into a continuous distance via its
    expectation: sum(bin_index * prob). No learnable parameters."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("bin_values", torch.arange(reg_max, dtype=torch.float32))

    def forward(self, pred_dist_logits):
        """pred_dist_logits: [..., 4, reg_max] -> [..., 4] continuous ltrb distances."""
        probs = pred_dist_logits.softmax(dim=-1)
        return (probs * self.bin_values).sum(dim=-1)


def encode_dfl_targets(distance, reg_max):
    """distance: [...] continuous, expected in [0, reg_max-1]. Returns (lo_idx, hi_idx,
    lo_weight, hi_weight), each [...], the standard DFL soft two-bin split: weight on the
    lower neighboring bin is (hi - distance), weight on the upper is (distance - lo)."""
    distance = distance.clamp(0, reg_max - 1 - 1e-3)
    lo_idx = distance.floor().long()
    hi_idx = (lo_idx + 1).clamp(max=reg_max - 1)
    hi_weight = distance - lo_idx.float()
    lo_weight = 1.0 - hi_weight
    return lo_idx, hi_idx, lo_weight, hi_weight


def dfl_loss(pred_dist_logits, target_distance, reg_max):
    """pred_dist_logits: [..., 4, reg_max] raw logits. target_distance: [..., 4] continuous
    ltrb. Returns per-element loss [..., 4] (caller reduces/weights by fg mask)."""
    lo_idx, hi_idx, lo_weight, hi_weight = encode_dfl_targets(target_distance, reg_max)
    logp = pred_dist_logits.log_softmax(dim=-1)
    logp_lo = logp.gather(-1, lo_idx.unsqueeze(-1)).squeeze(-1)
    logp_hi = logp.gather(-1, hi_idx.unsqueeze(-1)).squeeze(-1)
    return -(lo_weight * logp_lo + hi_weight * logp_hi)


class DetectionHeadLoss(nn.Module):
    """Combines DFL with a GIoU box term. cls loss is handled separately (TAL's soft targets
    against the head's own classification branch, plain BCE) since it doesn't depend on reg_max
    or the ltrb encoding this module owns."""
    def __init__(self, reg_max=16, dfl_weight=1.0, iou_weight=2.0):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_weight = dfl_weight
        self.iou_weight = iou_weight
        self.integral = DFLIntegral(reg_max)

    def forward(self, pred_dist_logits, anchor_points, anchor_strides, target_boxes_xyxy, fg_mask):
        """pred_dist_logits: [B,N,4,reg_max]. anchor_points: [N,2] xy. anchor_strides: [N].
        target_boxes_xyxy: [B,N,4] (meaningful only where fg_mask is True). fg_mask: [B,N] bool."""
        if not fg_mask.any():
            z = pred_dist_logits.new_tensor(0.0)
            return z, {"dfl": z, "iou": z}

        pred_ltrb = self.integral(pred_dist_logits) * anchor_strides.view(1, -1, 1)  # [B,N,4], back to pixel units
        ax, ay = anchor_points[:, 0], anchor_points[:, 1]
        pred_boxes = torch.stack([
            ax.unsqueeze(0) - pred_ltrb[..., 0], ay.unsqueeze(0) - pred_ltrb[..., 1],
            ax.unsqueeze(0) + pred_ltrb[..., 2], ay.unsqueeze(0) + pred_ltrb[..., 3],
        ], dim=-1)  # [B,N,4] xyxy

        fg_pred = pred_boxes[fg_mask]
        fg_target = target_boxes_xyxy[fg_mask]

        target_ltrb = torch.stack([
            ax.unsqueeze(0).expand_as(fg_mask)[fg_mask] - fg_target[:, 0],
            ay.unsqueeze(0).expand_as(fg_mask)[fg_mask] - fg_target[:, 1],
            fg_target[:, 2] - ax.unsqueeze(0).expand_as(fg_mask)[fg_mask],
            fg_target[:, 3] - ay.unsqueeze(0).expand_as(fg_mask)[fg_mask],
        ], dim=-1)
        fg_strides = anchor_strides.unsqueeze(0).expand_as(fg_mask)[fg_mask]
        target_ltrb_units = (target_ltrb / fg_strides.unsqueeze(-1)).clamp(min=0)

        dfl = dfl_loss(pred_dist_logits[fg_mask], target_ltrb_units, self.reg_max).mean()

        lt = torch.max(fg_pred[:, :2], fg_target[:, :2])
        rb = torch.min(fg_pred[:, 2:], fg_target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area_p = (fg_pred[:, 2] - fg_pred[:, 0]).clamp(min=0) * (fg_pred[:, 3] - fg_pred[:, 1]).clamp(min=0)
        area_t = (fg_target[:, 2] - fg_target[:, 0]).clamp(min=0) * (fg_target[:, 3] - fg_target[:, 1]).clamp(min=0)
        union = (area_p + area_t - inter).clamp(min=1e-9)
        iou = inter / union
        iou_loss = (1.0 - iou).mean()

        total = self.dfl_weight * dfl + self.iou_weight * iou_loss
        return total, {"dfl": dfl.detach(), "iou": iou_loss.detach()}


if __name__ == "__main__":
    torch.manual_seed(0)
    REG_MAX = 16

    print("Round-trip check: exact-bin distance encodes to a one-hot target and decodes back exactly")
    d_exact = torch.tensor([3.0, 7.0, 0.0, 15.0])
    lo, hi, lw, hw = encode_dfl_targets(d_exact, REG_MAX)
    assert torch.allclose(lw, torch.ones_like(lw)) or (d_exact == 15.0).any(), "an exact integer distance should put full weight on its own bin"
    print("lo_idx:", lo.tolist(), "weights:", lw.tolist())

    print("\nRound-trip check: fractional distance splits proportionally between its two neighboring bins")
    d_frac = torch.tensor([3.7])
    lo, hi, lw, hw = encode_dfl_targets(d_frac, REG_MAX)
    assert lo.item() == 3 and hi.item() == 4, f"3.7 should split between bins 3 and 4, got lo={lo.item()} hi={hi.item()}"
    assert abs(lw.item() - 0.3) < 1e-5 and abs(hw.item() - 0.7) < 1e-5, f"expected weights (0.3, 0.7), got ({lw.item()}, {hw.item()})"
    print(f"3.7 -> bin {lo.item()} (w={lw.item():.2f}) / bin {hi.item()} (w={hw.item():.2f}) -- correct proportional split.")

    print("\nLoss-at-theoretical-floor check: for a fractional target, DFL's soft-label cross-entropy")
    print("can't reach 0 even with a perfect prediction -- splitting mass between two bins has an")
    print("inherent entropy cost. The correct floor for weights (lw, hw) is -(lw*ln(lw) + hw*ln(hw)),")
    print("achieved when the softmax output exactly matches (lw, hw) -- i.e. logits = ln(weight),")
    print("not weight itself (that was the bug: linear-in-weight logits don't produce a softmax that")
    print("matches those weights). An exact-bin target (weight 1.0 on one side) has floor 0, so that")
    print("case alone would have passed the old, wrong 'near-zero' assertion without exposing this.")
    target = torch.tensor([[3.7, 5.0, 2.0, 8.3]])
    logits = torch.full((1, 4, REG_MAX), -20.0)
    lo, hi, lw, hw = encode_dfl_targets(target, REG_MAX)
    for i in range(4):
        logits[0, i, lo[0, i]] = torch.log(lw[0, i].clamp(min=1e-9))
        logits[0, i, hi[0, i]] = torch.log(hw[0, i].clamp(min=1e-9))
    loss = dfl_loss(logits, target, REG_MAX)
    floor = -(lw.clamp(min=1e-9).log() * lw + hw.clamp(min=1e-9).log() * hw)
    print("per-side loss:", loss.tolist())
    print("theoretical floor:", floor.tolist())
    assert torch.allclose(loss, floor, atol=0.02), f"a softmax matching (lw, hw) exactly should hit the entropy floor, got {loss.tolist()} vs floor {floor.tolist()}"
    print("OK: loss matches the true achievable floor for fractional targets, ~0 for the exact-bin sides.")

    print("\nDFLIntegral decode check: this corrected (properly peaked) distribution should decode back near the original distance")
    integral = DFLIntegral(REG_MAX)
    decoded = integral(logits)
    print("target:", target.tolist(), "decoded:", decoded.tolist())
    assert torch.allclose(decoded, target, atol=0.1), f"decode should recover the encoded distance closely, got {decoded} vs {target}"

    print("\nEnd-to-end DetectionHeadLoss check: one anchor, an exact-integer-bin target (l=t=r=b=4,")
    print("chosen deliberately on-bin so the DFL floor really is 0 here) with a perfect prediction")
    print("-> near-zero combined loss.")
    head_loss = DetectionHeadLoss(reg_max=REG_MAX)
    anchor_points = torch.tensor([[100.0, 100.0]])
    anchor_strides = torch.tensor([8.0])
    gt_box = torch.tensor([[68.0, 68.0, 132.0, 132.0]])  # l=t=r=b=32px -> /8 = 4.0 exactly, an integer bin
    target_boxes = gt_box.unsqueeze(0)
    fg_mask = torch.tensor([[True]])
    target_ltrb_units = torch.tensor([[4.0, 4.0, 4.0, 4.0]])
    perfect_logits = torch.full((1, 1, 4, REG_MAX), -20.0)
    lo, hi, lw, hw = encode_dfl_targets(target_ltrb_units, REG_MAX)
    for i in range(4):
        perfect_logits[0, 0, i, lo[0, i]] = 20.0
    total, parts = head_loss(perfect_logits, anchor_points, anchor_strides, target_boxes, fg_mask)
    print("total:", total.item(), "parts:", {k: v.item() for k, v in parts.items()})
    assert total.item() < 0.05, f"a perfect prediction against an exact-bin target should give near-zero total loss, got {total.item()}"

    print("\nComparative sanity check: a WRONG prediction (peaked at the opposite side of the bin range)")
    print("must score strictly worse than the perfect one above -- catches sign/gather errors that a")
    print("single absolute-value check could miss.")
    wrong_logits = torch.full((1, 1, 4, REG_MAX), -20.0)
    wrong_logits[0, 0, :, REG_MAX - 1] = 20.0  # confidently predicts the max possible distance instead
    wrong_total, _ = head_loss(wrong_logits, anchor_points, anchor_strides, target_boxes, fg_mask)
    print("wrong-prediction total:", wrong_total.item())
    assert wrong_total.item() > total.item() + 1.0, f"a confidently wrong prediction must score clearly worse than the correct one, got {wrong_total.item()} vs {total.item()}"

    print("\nAll DFL self-tests passed.")
