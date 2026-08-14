"""CrackDet detection heads (Sec. 3.2, Fig. 2c).

Fig. 2c shows, on top of the ReEDNet backbone output:
  - a shared `heatmap` branch and a shared `offset` branch (center-point
    detection, exactly as in vanilla CenterNet), and
  - 4 parallel per-angle-range branches, each producing `size` (h, w),
    `angle`, and `angle std` (the learned confidence used for variance
    voting at inference, Sec. 3.4).

The paper states each head is "simply implement[ed]... with a
fully-connected layer" on top of the shared backbone features. Sec. 3.2
also argues that both size and angle regression should benefit from
*rotation-equivariant* features (that's the whole point of ReEDNet), while
center-point detection (heatmap/offset) does not have an orientation of
its own. This module reflects that split explicitly:
  - heatmap/offset are computed from a rotation-*invariant* projection of
    the backbone features (`e2cnn.nn.GroupPooling`, i.e. max-over-the-group,
    the standard way to turn a regular-representation field into an
    invariant one),
  - each of the 4 branches first pushes the shared equivariant features
    through its own small equivariant conv (so the branch can specialize
    while remaining rotation-equivariant, matching the paper's stated
    rationale), and only *then* group-pools to a plain per-pixel feature
    vector, on top of which the literal "fully-connected"/1x1-conv
    regression heads (size, angle, angle-std) sit.

This branch-then-pool split is this module's own documented design choice
for the part of the architecture the paper leaves unspecified (there is no
released reference implementation) -- see the package README.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2cnn import nn as enn

from .piecewise_angle import NUM_BRANCHES


class CenterHead(nn.Module):
    """Shared heatmap + offset heads, operating on invariant (group-pooled) features."""

    def __init__(self, backbone_out_type, num_classes: int = 1, hidden_channels: int = 64):
        super().__init__()
        self.pool = enn.GroupPooling(backbone_out_type)
        in_channels = self.pool.out_type.size

        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        # CenterNet convention: heatmap head's final-layer bias initialized so the
        # network starts near a low foreground prior (avoids early focal-loss blowup).
        nn.init.constant_(self.heatmap[-1].bias, -2.19)

    def forward(self, feat: "enn.GeometricTensor"):
        inv = self.pool(feat).tensor
        return {"heatmap": self.heatmap(inv), "offset": self.offset(inv)}


class AngleBranchHead(nn.Module):
    """One of the 4 piecewise-angle branches: its own equivariant trunk + size/angle/std heads."""

    # Bounds on predicted sigma (see forward()) -- MAX_STD chosen generously relative to the
    # MAR loss's own "1/2" constant (Eq. 3): sigma^2 up to MAX_STD^2=100 already dwarfs 0.5,
    # so term1 is already fully saturated toward 1 long before this ceiling, while still
    # giving the "push other branches to high variance" term plenty of room to express real
    # ambiguity before hitting the cap.
    MIN_STD = 1e-3
    MAX_STD = 10.0

    def __init__(self, gspace, backbone_out_type, trunk_mult: int = 8, hidden_channels: int = 64):
        super().__init__()
        trunk_type = enn.FieldType(gspace, trunk_mult * [gspace.regular_repr])
        self.trunk_conv = enn.R2Conv(backbone_out_type, trunk_type, kernel_size=3, padding=1,
                                      bias=False)
        self.trunk_bn = enn.InnerBatchNorm(trunk_type)
        self.trunk_act = enn.ReLU(trunk_type, inplace=True)
        self.pool = enn.GroupPooling(trunk_type)
        in_channels = self.pool.out_type.size

        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.angle_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.std_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feat: "enn.GeometricTensor"):
        x = self.trunk_conv(feat)
        x = self.trunk_bn(x)
        x = self.trunk_act(x)
        inv = self.pool(x).tensor

        # h, w must stay positive (they're box side lengths, reprojected by a
        # bounded trig factor in [~0.707, 1] -- see piecewise_angle.py), so softplus.
        size = F.softplus(self.size_head(inv))
        # theta_i is a genuine regression target in degrees within this branch's
        # ~45-degree local range -- left unbounded, matching the paper's framing
        # of this as regression (not classification into sub-bins).
        angle = self.angle_head(inv)
        # sigma_i must be bounded on BOTH ends. The lower bound (unchanged from
        # before) keeps the MAR-loss denominator (1/2 + sigma_i^2) away from 0.
        # The upper bound is new, added after a real RunPod smoke test showed
        # training loss diverging to increasingly large negative values across
        # epochs -- traced to model/losses.py's MARLoss: its "-sum_other" term
        # (Eq. 3-4's own -sum_{j!=i} sigma_j^2) has no saturating denominator
        # and is subtracted raw, so nothing in the paper's stated loss stops a
        # non-valid branch's sigma from being driven toward infinity, which is
        # always "free" reward with zero counterforce. The paper doesn't say
        # what bounds this in practice; this is this implementation's own fix,
        # not something the paper specifies -- a sigmoid-scaled range instead
        # of unbounded softplus, so sigma can express "very uncertain" without
        # the network having anywhere to run away to.
        std = self.MIN_STD + (self.MAX_STD - self.MIN_STD) * torch.sigmoid(self.std_head(inv))
        return size, angle, std


class MultiBranchHead(nn.Module):
    """All 4 piecewise-angle branches, stacked along a new branch dimension."""

    def __init__(self, gspace, backbone_out_type, trunk_mult: int = 8, hidden_channels: int = 64):
        super().__init__()
        self.branches = nn.ModuleList([
            AngleBranchHead(gspace, backbone_out_type, trunk_mult, hidden_channels)
            for _ in range(NUM_BRANCHES)
        ])

    def forward(self, feat: "enn.GeometricTensor"):
        sizes, angles, stds = [], [], []
        for branch in self.branches:
            size, angle, std = branch(feat)
            sizes.append(size)
            angles.append(angle)
            stds.append(std)
        # Each stacked as (B, NUM_BRANCHES, C, H, W).
        return torch.stack(sizes, dim=1), torch.stack(angles, dim=1), torch.stack(stds, dim=1)
