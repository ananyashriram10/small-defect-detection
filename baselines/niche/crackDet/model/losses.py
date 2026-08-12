"""CrackDet training losses: L_k, L_off, L_size, and the MAR loss (Sec. 3.2-3.3, Eq. 3-5).

    L_CrackDet = L_k + lambda_off * L_off + lambda_size * L_size + lambda_MAR * L_MAR   (Eq. 5)

with lambda_off=0.1, lambda_size=0.2, lambda_MAR=0.1 (Sec. 4.1, "According
to our hyper-parameter sensitivity study").

- L_k: the standard CenterNet penalty-reduced pixelwise logistic (focal)
  loss on the heatmap (paper: "Lk and Loff are the losses of center point
  recognition and offset regression by following CenterNet [73]").
- L_off: L1 offset regression at ground-truth center locations.
- L_size: L1 on the *branch-local* redefined (h_i, w_i), Sec. 3.3:
  "Lsize = |h_i_hat - h_i| + |w_i_hat - w_i|", supervised only on the
  branch whose valid range contains the ground-truth angle.
- L_MAR (Eq. 3-4): a Wasserstein-distance-based loss between the predicted
  per-branch angle distribution (mean theta_i_hat, std sigma_i) and the
  near-delta ground-truth distribution, on the valid branch, *combined*
  with a term that rewards the other 3 (non-valid) branches for predicting
  large variance at that same location -- this is what lets variance
  voting (postprocess.py) suppress the wrong branches at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# L_k: CenterNet penalty-reduced focal loss on the heatmap.
# ---------------------------------------------------------------------------

class GaussianFocalLoss(nn.Module):
    """CenterNet's Eq. for L_k (Zhou et al., "Objects as Points", the paper's ref [73]).

    `pred` is raw logits (B, C, H, W); `gt` is the rendered Gaussian
    heatmap in [0, 1] with the same shape (1.0 at true centers, decaying
    Gaussian bumps elsewhere, 0 in the background -- see
    `data/target_generator.py`).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred_logits).clamp(min=self.eps, max=1 - self.eps)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = torch.log(prob) * torch.pow(1 - prob, self.alpha) * pos_mask
        neg_loss = torch.log(1 - prob) * torch.pow(prob, self.alpha) * neg_weights * neg_mask

        num_pos = pos_mask.sum().clamp(min=1.0)
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos


# ---------------------------------------------------------------------------
# Gather utility: pull per-branch, per-pixel predictions out at GT center locations.
# ---------------------------------------------------------------------------

def gather_at_centers(pred: torch.Tensor, batch_idx: torch.Tensor, pixel_idx: torch.Tensor):
    """Index a (B, ..., H, W) prediction tensor at K (batch, flat-pixel) locations.

    `pred` may have any number of leading dims after batch (e.g. plain
    (B, C, H, W) for offset/heatmap, or (B, NUM_BRANCHES, C, H, W) for the
    per-branch size/angle/std heads) -- everything except the last two
    (H, W) dims is treated as "channels" and gathered along with the pixel.

    Returns a tensor of shape (K, *channel_dims).
    """
    b, *chan_dims, h, w = pred.shape
    flat = pred.reshape(b, -1, h * w)                      # (B, C_total, H*W)
    flat = flat.permute(0, 2, 1)                            # (B, H*W, C_total)
    gathered = flat[batch_idx, pixel_idx]                   # (K, C_total)
    return gathered.reshape(len(batch_idx), *chan_dims)


class RegL1Loss(nn.Module):
    """L1 regression loss evaluated only at ground-truth positive locations."""

    def forward(self, pred: torch.Tensor, batch_idx: torch.Tensor, pixel_idx: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if len(batch_idx) == 0:
            return pred.sum() * 0.0
        gathered = gather_at_centers(pred, batch_idx, pixel_idx)
        return F.l1_loss(gathered, target, reduction="mean")


# ---------------------------------------------------------------------------
# L_MAR: Eq. 3-4.
# ---------------------------------------------------------------------------

class MARLoss(nn.Module):
    """Multi-branch Angle Regression loss (Sec. 3.3, Eq. 3-4).

    For a ground-truth box whose angle falls in branch i (branch_idx),
    with redefined local ground-truth angle theta_i:

        L_MAR = (||theta_i_hat - theta_i||^2 + sigma_i^2) / (1/2 + sigma_i^2)
                - sum_{j != i} sigma_j^2

    theta_i_hat, sigma_i are the *valid* branch's predictions at the GT
    center pixel; sigma_j are the *other 3* branches' predicted stds at
    that same pixel. Minimizing this jointly (a) pulls theta_i_hat toward
    theta_i with a Wasserstein-distance term that degrades gracefully
    instead of blowing up when sigma_i grows (used for genuinely ambiguous
    sub-crack orientations, Fig. 1c), and (b) pushes the non-matching
    branches to report high uncertainty at that pixel, which is exactly
    what `postprocess.py`'s variance voting relies on at inference.
    """

    def forward(self, pred_angle: torch.Tensor, pred_std: torch.Tensor,
                batch_idx: torch.Tensor, pixel_idx: torch.Tensor,
                branch_idx: torch.Tensor, gt_theta_i: torch.Tensor) -> torch.Tensor:
        if len(batch_idx) == 0:
            return pred_angle.sum() * 0.0

        # (K, NUM_BRANCHES) after gathering the single angle/std channel per branch.
        angle_all = gather_at_centers(pred_angle, batch_idx, pixel_idx).squeeze(-1)
        std_all = gather_at_centers(pred_std, batch_idx, pixel_idx).squeeze(-1)
        sigma2_all = std_all.pow(2)

        rows = torch.arange(len(batch_idx), device=pred_angle.device)
        theta_hat = angle_all[rows, branch_idx]
        sigma2_valid = sigma2_all[rows, branch_idx]

        wasserstein = (theta_hat - gt_theta_i).pow(2) + sigma2_valid
        term1 = wasserstein / (0.5 + sigma2_valid)

        sum_other = sigma2_all.sum(dim=1) - sigma2_valid
        per_sample = term1 - sum_other
        return per_sample.mean()


# ---------------------------------------------------------------------------
# Full training objective: Eq. 5.
# ---------------------------------------------------------------------------

class CrackDetLoss(nn.Module):
    def __init__(self, lambda_off: float = 0.1, lambda_size: float = 0.2,
                 lambda_mar: float = 0.1):
        super().__init__()
        self.lambda_off = lambda_off
        self.lambda_size = lambda_size
        self.lambda_mar = lambda_mar
        self.heatmap_loss = GaussianFocalLoss()
        self.reg_l1 = RegL1Loss()
        self.mar_loss = MARLoss()

    def forward(self, preds: dict, targets: dict) -> dict:
        """`preds` is a CrackDet.forward() output dict. `targets` is produced by
        `data/target_generator.py` and must contain:
            heatmap        (B, num_classes, H, W)
            offset         (K, 2)
            size           (K, 2)   redefined (h_i, w_i) for the valid branch
            theta_i        (K,)     redefined local angle for the valid branch
            batch_idx      (K,)
            pixel_idx      (K,)     flattened y*W + x
            branch_idx     (K,)
        """
        l_k = self.heatmap_loss(preds["heatmap"], targets["heatmap"])
        l_off = self.reg_l1(preds["offset"], targets["batch_idx"], targets["pixel_idx"],
                             targets["offset"])

        # size/angle/std predictions carry a branch dim (B, NUM_BRANCHES, C, H, W); gather the
        # valid branch's (h_i, w_i) explicitly rather than reusing RegL1Loss's generic gather.
        size_all = gather_at_centers(preds["size"], targets["batch_idx"], targets["pixel_idx"])
        if len(targets["batch_idx"]) > 0:
            rows = torch.arange(len(targets["batch_idx"]), device=size_all.device)
            size_valid = size_all[rows, targets["branch_idx"]]
            l_size = F.l1_loss(size_valid, targets["size"], reduction="mean")
        else:
            l_size = preds["size"].sum() * 0.0

        l_mar = self.mar_loss(preds["angle"], preds["std"], targets["batch_idx"],
                               targets["pixel_idx"], targets["branch_idx"], targets["theta_i"])

        total = l_k + self.lambda_off * l_off + self.lambda_size * l_size + self.lambda_mar * l_mar
        return {"loss": total, "loss_heatmap": l_k, "loss_offset": l_off,
                "loss_size": l_size, "loss_mar": l_mar}
