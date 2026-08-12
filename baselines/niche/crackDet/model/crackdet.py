"""CrackDet: the full oriented sub-crack detector (Fig. 2c end-to-end).

Assembles the rotation-equivariant ReEDNet backbone (`backbone.py`) with
the shared center-point heads and the 4 piecewise-angle branch heads
(`heads.py`) into the model described in Sec. 3.2-3.4.

Output dict, all at stride 4 relative to the input image:
    heatmap : (B, num_classes, H/4, W/4)   raw logits (sigmoid applied by the loss/decoder)
    offset  : (B, 2,           H/4, W/4)   sub-pixel center offset (dx, dy)
    size    : (B, 4, 2,        H/4, W/4)   per-branch (h_i, w_i), branch dim = 4
    angle   : (B, 4, 1,        H/4, W/4)   per-branch local angle theta_i (degrees)
    std     : (B, 4, 1,        H/4, W/4)   per-branch angle std sigma_i (> 0)

`4` is `piecewise_angle.NUM_BRANCHES`; the branch dim always comes right
after the batch dim so `size`/`angle`/`std` can be indexed/gathered by
branch consistently with the targets produced in `data/target_generator.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import ReEDNet
from .heads import CenterHead, MultiBranchHead
from .piecewise_angle import NUM_BRANCHES


class CrackDet(nn.Module):
    def __init__(self, num_classes: int = 1, group_order: int = 8, branch_trunk_mult: int = 8,
                 head_hidden_channels: int = 64):
        super().__init__()
        self.backbone = ReEDNet(in_channels=3, group_order=group_order)
        self.center_head = CenterHead(self.backbone.out_type, num_classes=num_classes,
                                       hidden_channels=head_hidden_channels)
        self.branch_head = MultiBranchHead(self.backbone.gspace, self.backbone.out_type,
                                            trunk_mult=branch_trunk_mult,
                                            hidden_channels=head_hidden_channels)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        center_out = self.center_head(feat)
        size, angle, std = self.branch_head(feat)
        return {
            "heatmap": center_out["heatmap"],
            "offset": center_out["offset"],
            "size": size,
            "angle": angle,
            "std": std,
        }

    @property
    def output_stride(self) -> int:
        return 4

    @property
    def num_branches(self) -> int:
        return NUM_BRANCHES
