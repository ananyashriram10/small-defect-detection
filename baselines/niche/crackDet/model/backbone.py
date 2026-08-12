"""ReEDNet: the rotation-equivariant encoder-decoder backbone from CrackDet (Sec. 3.2).

From the paper (Sec. 3.2, "Rotation-equivariant backbone"): "We first adopt
CenterNet as our baseline... Then, we re-implement all layers of the
fully-convolutional encoder-decoder networks (i.e., up-convolutional
residual networks) in CenterNet based on e2cnn, named as ReEDNet... Thanks
to the rotation weight sharing and group representations in e2cnn, our
ReEDNet takes smaller parameters than the original encoder-decoder
networks in CenterNet and enjoys the capability of equivariance."

Table 4 additionally labels the backbone used for all reported results as
"ReED-R-50", i.e. built on a ResNet-50 layout: Bottleneck blocks
(expansion 4) with stage depths [3, 4, 6, 3] and stage widths
[64, 128, 256, 512] (pre-expansion), exactly like torchvision's ResNet-50 --
just re-expressed with e2cnn equivariant layers (regular representations
of the cyclic rotation group C_N) instead of plain nn.Conv2d. The decoder
mirrors CenterNet's standard ResNet-variant head: 3 transpose-conv-style
upsampling stages (256 -> 128 -> 64 raw channels) taking the stride-32
bottleneck output back up to stride-4, which is where CenterNet's
heatmap/offset/size heads are normally attached.

What the paper does NOT specify (the official implementation was never
released), and where this module makes an explicit, documented choice:
  - The rotation group order N. We use N=8 (C_8), matching the group order
    used by ReDet's ReResNet (Han et al., CVPR'21, ref [16] in the paper) --
    the closest publicly-documented e2cnn-based equivariant ResNet-50 for
    oriented object detection, and the natural reference point since the
    paper explicitly contrasts itself with ReDet.
  - Exact channel-to-multiplicity mapping. A raw channel count C is
    represented as `C // N` copies of the group's regular representation
    (each regular_repr contributes N raw channels), so the *raw* tensor
    channel counts at every stage match a real ResNet-50 exactly
    (64/256/512/1024/2048), while internally each is carried as an N-way
    rotation-equivariant feature field.
  - Whether the decoder fuses encoder skip connections. The original
    CenterNet ResNet variant (and the paper's Fig. 2c) shows a plain
    deconv stack with no U-Net-style skip fusion, so none is added here.

This is a from-spec reimplementation, not a port of an official codebase
(none exists publicly for this paper) -- see the package README for the
full faithfulness/limitations notes.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

try:
    from e2cnn import gspaces
    from e2cnn import nn as enn
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "ReEDNet requires the 'e2cnn' package (pip install e2cnn) -- the "
        "E(2)-equivariant CNN library the CrackDet paper explicitly builds "
        "its rotation-equivariant backbone on (Sec. 3.2)."
    ) from exc


def _field(gspace, multiplicity: int) -> "enn.FieldType":
    """`multiplicity` copies of the regular representation of the rotation group."""
    return enn.FieldType(gspace, multiplicity * [gspace.regular_repr])


class _EquivConvBNAct(nn.Module):
    """R2Conv -> InnerBatchNorm -> (optional) ReLU, the basic equivariant unit."""

    def __init__(self, in_type, out_type, kernel_size, stride=1, padding=None, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=False)
        self.bn = enn.InnerBatchNorm(out_type)
        self.act = enn.ReLU(out_type, inplace=True) if act else None
        self.out_type = out_type

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class _EquivBottleneck(nn.Module):
    """Equivariant ResNet-50-style bottleneck block (1x1 reduce, 3x3, 1x1 expand x4)."""

    expansion = 4

    def __init__(self, gspace, in_type, mid_mult: int, stride: int = 1):
        super().__init__()
        mid_type = _field(gspace, mid_mult)
        out_type = _field(gspace, mid_mult * self.expansion)

        self.conv1 = _EquivConvBNAct(in_type, mid_type, kernel_size=1, act=True)
        self.conv2 = _EquivConvBNAct(mid_type, mid_type, kernel_size=3, stride=stride, act=True)
        self.conv3 = _EquivConvBNAct(mid_type, out_type, kernel_size=1, act=False)

        self.needs_shortcut = (stride != 1) or (in_type != out_type)
        if self.needs_shortcut:
            self.shortcut = _EquivConvBNAct(in_type, out_type, kernel_size=1, stride=stride,
                                             act=False)
        else:
            self.shortcut = None
        self.final_act = enn.ReLU(out_type, inplace=True)
        self.out_type = out_type

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        return self.final_act(out)


class _EquivUpBlock(nn.Module):
    """2x bilinear-equivariant upsample, then a 3x3 conv to change the field type.

    Mirrors CenterNet's deconv decoder stage (Objects as Points, Sec. 4 /
    Huang et al. speed-accuracy trade-offs, ref [20]): upsample spatially,
    then a conv to reduce channel width, then BN+ReLU.
    """

    def __init__(self, in_type, out_type):
        super().__init__()
        self.upsample = enn.R2Upsampling(in_type, scale_factor=2, mode="bilinear",
                                          align_corners=False)
        self.conv = _EquivConvBNAct(in_type, out_type, kernel_size=3, act=True)
        self.out_type = out_type

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class ReEDNet(nn.Module):
    """Rotation-equivariant encoder-decoder backbone (ResNet-50 layout, e2cnn C_N group).

    Input: a standard image tensor, shape (B, 3, H, W), H and W multiples of 32.
    Output: an `e2cnn.nn.GeometricTensor` at stride 4 (H/4, W/4), field type
    `self.out_type` (regular representation, multiplicity `out_multiplicity`,
    so `out_multiplicity * group_order` raw channels) -- this feeds the
    multi-branch heads in `heads.py`.
    """

    STAGE_DEPTHS = (3, 4, 6, 3)          # ResNet-50 layer1..4 block counts
    STAGE_MID_MULT_BASE = (64, 128, 256, 512)  # pre-expansion widths, ResNet-50

    def __init__(self, in_channels: int = 3, group_order: int = 8):
        super().__init__()
        self.group_order = group_order
        self.gspace = gspaces.Rot2dOnR2(N=group_order)

        self.input_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        assert 64 % group_order == 0, (
            "group_order must divide every ResNet-50 stage width (64/128/256/512) so raw "
            "channel counts come out exact; N=8 (the default) satisfies this."
        )
        stem_mult = 64 // group_order
        stem_type = _field(self.gspace, stem_mult)

        # Stem: 7x7 stride-2 conv + 3x3 stride-2 maxpool -> stride 4, matches torchvision ResNet stem.
        self.stem_conv = _EquivConvBNAct(self.input_type, stem_type, kernel_size=7, stride=2,
                                          padding=3, act=True)
        self.stem_pool = enn.PointwiseMaxPool(stem_type, kernel_size=3, stride=2, padding=1)

        mid_mults = [w // group_order for w in self.STAGE_MID_MULT_BASE]
        assert all(m > 0 for m in mid_mults), "group_order too large for these stage widths"

        self.layer1 = self._make_stage(stem_type, mid_mults[0], self.STAGE_DEPTHS[0], stride=1)
        self.layer2 = self._make_stage(self.layer1[-1].out_type, mid_mults[1],
                                        self.STAGE_DEPTHS[1], stride=2)
        self.layer3 = self._make_stage(self.layer2[-1].out_type, mid_mults[2],
                                        self.STAGE_DEPTHS[2], stride=2)
        self.layer4 = self._make_stage(self.layer3[-1].out_type, mid_mults[3],
                                        self.STAGE_DEPTHS[3], stride=2)

        # Decoder: stride 32 -> 16 -> 8 -> 4, raw channel widths 256 -> 128 -> 64 (CenterNet's
        # standard ResNet-variant deconv head).
        dec_mults = [256 // group_order, 128 // group_order, 64 // group_order]
        self.decoder3 = _EquivUpBlock(self.layer4[-1].out_type, _field(self.gspace, dec_mults[0]))
        self.decoder2 = _EquivUpBlock(self.decoder3.out_type, _field(self.gspace, dec_mults[1]))
        self.decoder1 = _EquivUpBlock(self.decoder2.out_type, _field(self.gspace, dec_mults[2]))

        self.out_type = self.decoder1.out_type
        self.out_multiplicity = dec_mults[2]

    def _make_stage(self, in_type, mid_mult: int, depth: int, stride: int) -> nn.ModuleList:
        blocks: List[_EquivBottleneck] = []
        blocks.append(_EquivBottleneck(self.gspace, in_type, mid_mult, stride=stride))
        for _ in range(depth - 1):
            blocks.append(_EquivBottleneck(self.gspace, blocks[-1].out_type, mid_mult, stride=1))
        return nn.ModuleList(blocks)

    def _run_stage(self, stage: nn.ModuleList, x):
        for block in stage:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> "enn.GeometricTensor":
        x = enn.GeometricTensor(x, self.input_type)
        x = self.stem_conv(x)
        x = self.stem_pool(x)
        x = self._run_stage(self.layer1, x)      # stride 4
        x = self._run_stage(self.layer2, x)      # stride 8
        x = self._run_stage(self.layer3, x)      # stride 16
        x = self._run_stage(self.layer4, x)      # stride 32
        x = self.decoder3(x)                     # stride 16
        x = self.decoder2(x)                     # stride 8
        x = self.decoder1(x)                     # stride 4
        return x
