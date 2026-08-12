"""Inference-time decoding: heatmap peak extraction + angle-based variance voting (Sec. 3.4, Fig. 4).

From the paper: "Firstly, we extract the peaks in the heatmap by following
CenterNet. Then, we get standard deviations sigma_1..sigma_4 from four
branches at each peak. ... for each peak, we choose branch i that has the
minimum estimated variance ... i* = argmin_i sigma_i^2. Next, ... we obtain
the center (x, y) of the oriented bounding box according to the above
peaks and the offset branch. Finally, we acquire the final height
ho = h_i*_hat / Delta(theta_i*_hat), width wo = w_i*_hat / Delta(theta_i*_hat),
and angle theta_o = Gamma(theta_i*_hat)."

All spatial quantities (center, h, w) are computed and returned in
*feature-map* (stride-4) units; multiply by `stride` to get image-pixel
units (done here via the `stride` argument so callers get image-scale
boxes directly).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .piecewise_angle import decode_box


def _nms_peaks(heatmap_prob: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only local-maxima pixels (CenterNet-style max-pool NMS), zero out the rest."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap_prob, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == heatmap_prob).float()
    return heatmap_prob * keep


def _topk(heatmap_prob: torch.Tensor, k: int):
    """Top-k peaks per image, across all classes.

    Returns scores, classes, y, x (each shape (B, k)).
    """
    b, c, h, w = heatmap_prob.shape
    topk_per_class_scores, topk_per_class_inds = torch.topk(heatmap_prob.view(b, c, -1),
                                                              min(k, h * w))
    topk_per_class_ys = torch.div(topk_per_class_inds, w, rounding_mode="floor").float()
    topk_per_class_xs = (topk_per_class_inds % w).float()

    k1 = topk_per_class_scores.size(-1)
    scores, inds = torch.topk(topk_per_class_scores.view(b, -1), min(k, c * k1))
    classes = torch.div(inds, k1, rounding_mode="floor").long()
    inds = inds % k1

    def _gather(x):
        return torch.gather(x.view(b, -1), 1, (classes * k1 + inds))

    ys = _gather(topk_per_class_ys)
    xs = _gather(topk_per_class_xs)
    return scores, classes, ys, xs


@torch.no_grad()
def decode(preds: dict, topk: int = 100, score_thresh: float = 0.1, nms_kernel: int = 3,
           stride: int = 4):
    """Decode a CrackDet forward-pass output dict into oriented boxes.

    Returns a list (length B) of dicts, one per image, each with tensors:
        boxes  : (N, 5) -> (cx, cy, h, w, theta_deg), image-pixel scale
        scores : (N,)   heatmap confidence
        sigma  : (N,)   winning branch's angle std (lower = more confident orientation)
        branch : (N,)   winning branch index in {0,1,2,3}
    N <= topk after score-thresholding.
    """
    heatmap_prob = torch.sigmoid(preds["heatmap"])
    heatmap_prob = _nms_peaks(heatmap_prob, kernel=nms_kernel)
    scores, classes, ys, xs = _topk(heatmap_prob, topk)

    b, _, h, w = preds["heatmap"].shape
    device = preds["heatmap"].device
    batch_idx = torch.arange(b, device=device).view(b, 1).expand(b, ys.size(1)).reshape(-1)
    pixel_idx = (ys.long() * w + xs.long()).reshape(-1)
    scores_flat = scores.reshape(-1)
    classes_flat = classes.reshape(-1)
    xs_flat, ys_flat = xs.reshape(-1), ys.reshape(-1)

    keep = scores_flat > score_thresh
    if keep.sum() == 0:
        return [{"boxes": torch.zeros((0, 5), device=device), "scores": torch.zeros((0,), device=device),
                  "sigma": torch.zeros((0,), device=device), "branch": torch.zeros((0,), dtype=torch.long, device=device)}
                for _ in range(b)]

    batch_idx, pixel_idx = batch_idx[keep], pixel_idx[keep]
    scores_flat, classes_flat = scores_flat[keep], classes_flat[keep]
    xs_flat, ys_flat = xs_flat[keep], ys_flat[keep]

    # Offset refinement (B, 2, H, W) -> (K, 2).
    offset = preds["offset"].permute(0, 2, 3, 1).reshape(b, h * w, 2)
    off_kw = offset[batch_idx, pixel_idx]
    cx = xs_flat + off_kw[:, 0]
    cy = ys_flat + off_kw[:, 1]

    # Per-branch size/angle/std at these pixels: (B, NUM_BRANCHES, C, H, W) -> (K, NUM_BRANCHES, C).
    def gather_branchwise(t):
        bb, nb, cc, hh, ww = t.shape
        flat = t.reshape(bb, nb, cc, hh * ww).permute(0, 3, 1, 2)  # (B, H*W, NUM_BRANCHES, C)
        return flat[batch_idx, pixel_idx]                          # (K, NUM_BRANCHES, C)

    size_all = gather_branchwise(preds["size"])     # (K, 4, 2)
    angle_all = gather_branchwise(preds["angle"]).squeeze(-1)  # (K, 4)
    std_all = gather_branchwise(preds["std"]).squeeze(-1)      # (K, 4)

    # Variance voting: pick the branch with the smallest predicted std at each peak (Fig. 4).
    branch_star = std_all.argmin(dim=1)
    rows = torch.arange(len(branch_star), device=device)
    h_i = size_all[rows, branch_star, 0]
    w_i = size_all[rows, branch_star, 1]
    theta_i_hat = angle_all[rows, branch_star]
    sigma_star = std_all[rows, branch_star]

    h_o, w_o, theta_o = decode_box(h_i, w_i, theta_i_hat, branch_star)

    boxes = torch.stack([cx, cy, h_o, w_o, theta_o], dim=1) * torch.tensor(
        [stride, stride, stride, stride, 1.0], device=device)

    outputs = []
    for i in range(b):
        m = batch_idx == i
        outputs.append({
            "boxes": boxes[m],
            "scores": scores_flat[m],
            "sigma": sigma_star[m],
            "branch": branch_star[m],
            "class": classes_flat[m],
        })
    return outputs
