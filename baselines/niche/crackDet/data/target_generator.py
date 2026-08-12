"""Turn oriented sub-crack box annotations into CrackDet training targets.

Ground-truth boxes are given per image as (cx, cy, h, w, theta_deg) in
*image-pixel* coordinates, theta_deg in [0, 180). This module:
  1. Renders the CenterNet-style Gaussian heatmap at the model's output
     stride (paper: "Lk ... by following CenterNet [73]" -- the render
     radius / unbiased-Gaussian utilities here are the standard CornerNet/
     CenterNet formulas (Law & Deng, ECCV'18; Zhou et al. 2019's official
     `utils/image.py`), not something this paper redefines).
  2. For every box, determines its piecewise-angle branch and the
     branch-local redefined (theta_i, h_i, w_i) via
     `model.piecewise_angle.forward_transform` (Sec. 3.1).
  3. Emits per-instance regression targets (center sub-pixel offset,
     redefined size, redefined angle, branch index, flattened pixel
     index) for `model.losses.CrackDetLoss` to gather predictions against.

All box sizes here are handled in *feature-map* (stride-divided) units,
matching what the model regresses and what `model/postprocess.py` expects
before its own final `* stride` rescale back to image pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch

from model.piecewise_angle import forward_transform


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Standard CornerNet/CenterNet Gaussian radius so a heatmap bump at this
    radius has >= `min_overlap` IoU with the true box under any of 3 corner-
    displacement cases (Law & Deng, ECCV'18, Sec 3.1)."""
    a1, b1 = 1.0, (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))) / 2

    return max(min(r1, r2, r3), 0.0)


def _gaussian2d(shape, sigma: float) -> np.ndarray:
    m, n = [(s - 1.0) / 2.0 for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g


def draw_gaussian(heatmap: np.ndarray, center_xy, radius: float, k: float = 1.0) -> None:
    radius = max(int(round(radius)), 0)
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center_xy[0]), int(center_xy[1])
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if right <= -left or bottom <= -top:
        return

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


@dataclass
class OrientedBox:
    cx: float
    cy: float
    h: float
    w: float
    theta_deg: float
    label: int = 0


@dataclass
class CrackDetTargets:
    heatmap: torch.Tensor      # (num_classes, Hf, Wf)
    offset: torch.Tensor       # (K, 2)
    size: torch.Tensor         # (K, 2)  redefined (h_i, w_i), feature-map units
    theta_i: torch.Tensor      # (K,)    redefined local angle, degrees
    branch_idx: torch.Tensor   # (K,)    long
    pixel_idx: torch.Tensor    # (K,)    long, flattened y*Wf + x
    num_instances: int = field(init=False)

    def __post_init__(self):
        self.num_instances = int(self.branch_idx.numel())


class CrackDetTargetGenerator:
    def __init__(self, stride: int = 4, num_classes: int = 1, min_overlap: float = 0.7):
        self.stride = stride
        self.num_classes = num_classes
        self.min_overlap = min_overlap

    def __call__(self, image_hw, boxes: Sequence[OrientedBox]) -> CrackDetTargets:
        img_h, img_w = image_hw
        assert img_h % self.stride == 0 and img_w % self.stride == 0, (
            f"image size {(img_h, img_w)} must be a multiple of stride {self.stride}"
        )
        fh, fw = img_h // self.stride, img_w // self.stride

        heatmap = np.zeros((self.num_classes, fh, fw), dtype=np.float32)
        offsets: List[List[float]] = []
        sizes_img_scale: List[List[float]] = []
        thetas: List[float] = []
        pixel_idx: List[int] = []
        labels_for_boxes: List[int] = []

        for box in boxes:
            fcx, fcy = box.cx / self.stride, box.cy / self.stride
            fh_box, fw_box = box.h / self.stride, box.w / self.stride
            if fh_box <= 0 or fw_box <= 0:
                continue

            ix, iy = int(fcx), int(fcy)
            if not (0 <= ix < fw and 0 <= iy < fh):
                continue

            radius = gaussian_radius(fh_box, fw_box, self.min_overlap)
            draw_gaussian(heatmap[box.label], (ix, iy), radius)

            offsets.append([fcx - ix, fcy - iy])
            sizes_img_scale.append([fh_box, fw_box])
            thetas.append(box.theta_deg)
            pixel_idx.append(iy * fw + ix)
            labels_for_boxes.append(box.label)

        heatmap[np.isnan(heatmap)] = 0.0
        heatmap_t = torch.from_numpy(heatmap)

        if len(thetas) == 0:
            return CrackDetTargets(
                heatmap=heatmap_t,
                offset=torch.zeros((0, 2)),
                size=torch.zeros((0, 2)),
                theta_i=torch.zeros((0,)),
                branch_idx=torch.zeros((0,), dtype=torch.long),
                pixel_idx=torch.zeros((0,), dtype=torch.long),
            )

        theta_deg_t = torch.tensor(thetas, dtype=torch.float32).clamp(min=0.0, max=180.0 - 1e-4)
        sizes_t = torch.tensor(sizes_img_scale, dtype=torch.float32)
        theta_i, size_i_h, size_i_w, branch_idx = forward_transform(
            theta_deg_t, sizes_t[:, 0], sizes_t[:, 1]
        )

        return CrackDetTargets(
            heatmap=heatmap_t,
            offset=torch.tensor(offsets, dtype=torch.float32),
            size=torch.stack([size_i_h, size_i_w], dim=1),
            theta_i=theta_i,
            branch_idx=branch_idx,
            pixel_idx=torch.tensor(pixel_idx, dtype=torch.long),
        )


def collate_targets(per_image_targets: Sequence[CrackDetTargets]) -> dict:
    """Stack a list of per-image CrackDetTargets into one batched targets dict
    matching what `model.losses.CrackDetLoss.forward` expects."""
    heatmap = torch.stack([t.heatmap for t in per_image_targets], dim=0)

    batch_idx, offset, size, theta_i, branch_idx, pixel_idx = [], [], [], [], [], []
    for b, t in enumerate(per_image_targets):
        if t.num_instances == 0:
            continue
        batch_idx.append(torch.full((t.num_instances,), b, dtype=torch.long))
        offset.append(t.offset)
        size.append(t.size)
        theta_i.append(t.theta_i)
        branch_idx.append(t.branch_idx)
        pixel_idx.append(t.pixel_idx)

    def _cat(chunks, empty_shape):
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(empty_shape)

    return {
        "heatmap": heatmap,
        "batch_idx": _cat(batch_idx, (0,)).long(),
        "pixel_idx": _cat(pixel_idx, (0,)).long(),
        "offset": _cat(offset, (0, 2)),
        "size": _cat(size, (0, 2)),
        "theta_i": _cat(theta_i, (0,)),
        "branch_idx": _cat(branch_idx, (0,)).long(),
    }
