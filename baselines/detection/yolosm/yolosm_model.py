"""YOLO-SM model components.

The paper describes DCMNet + GMF + a decoupled anchor-free YOLO head. This
module keeps those pieces explicit so they can be inspected and ablated.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = HSigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, g: int = 1,
                 act: str = "hswish") -> None:
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = HSwish() if act == "hswish" else nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = ((k - 1) // 2) * dilation
        self.dw = nn.Conv2d(c1, c1, k, s, padding, dilation=dilation, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        self.dw_act = HSwish()
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.pw_act = HSwish()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_act(self.dw_bn(self.dw(x)))
        return self.pw_act(self.pw_bn(self.pw(x)))


class MobileBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, expansion: int = 4, stride: int = 1) -> None:
        super().__init__()
        hidden = max(c1, c1 * expansion)
        self.use_shortcut = stride == 1 and c1 == c2
        self.expand = ConvBNAct(c1, hidden, 1, act="hswish")
        self.depthwise = ConvBNAct(hidden, hidden, 3, stride, g=hidden, act="hswish")
        self.project = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.project(self.depthwise(self.expand(x)))
        return x + y if self.use_shortcut else y


class DCM(nn.Module):
    """Densely Connected Multi-scale module with dilation 7, 5, 3, 1."""

    def __init__(self, channels: int, dilations: Sequence[int] = (7, 5, 3, 1)) -> None:
        super().__init__()
        self.compress = nn.ModuleList(
            [ConvBNAct(channels * (i + 1), channels, 1, act="hswish") for i in range(len(dilations))]
        )
        self.branches = nn.ModuleList(
            [DepthwiseSeparableConv(channels, channels, 3, 1, d) for d in dilations]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (len(dilations) + 1), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            HSwish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for compress, branch in zip(self.compress, self.branches):
            dense_input = compress(torch.cat(features, dim=1))
            features.append(branch(dense_input))
        return self.fuse(torch.cat(features, dim=1)) + x


class SAM(nn.Module):
    """Stereoscopic Attention Mechanism from the paper.

    This is a small channel/spatial attention block, not Segment Anything.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        kernel = int(abs((math.log2(max(channels, 1)) + b) / gamma))
        kernel = kernel if kernel % 2 == 1 else kernel + 1
        kernel = max(kernel, 3)
        self.channel_conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, 1, bias=False)
        self.gate = HSigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        mx = F.adaptive_max_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        channel = self.gate(self.channel_conv(avg) + self.channel_conv(mx))
        channel = channel.transpose(1, 2).unsqueeze(-1)

        spatial_avg = x.mean(dim=1, keepdim=True)
        spatial_max = x.amax(dim=1, keepdim=True)
        spatial = self.gate(self.spatial_conv(torch.cat([spatial_max, spatial_avg], dim=1)))
        return x * channel * spatial


class MCDownsample(nn.Module):
    """Max-pooling plus convolutional downsampling module."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        mid = max(c2 // 2, 8)
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBNAct(c1, mid, 1),
        )
        self.conv_branch = nn.Sequential(
            ConvBNAct(c1, mid, 1),
            ConvBNAct(mid, c2 - mid, 3, 2),
        )
        self.fuse = ConvBNAct(c2, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(torch.cat([self.pool_branch(x), self.conv_branch(x)], dim=1))


class GSConv2D(nn.Module):
    """Ghost shuffle convolution used by the GMF neck."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        primary = max(c2 // 2, 1)
        ghost = c2 - primary
        self.primary = ConvBNAct(c1, primary, k, s)
        self.cheap = ConvBNAct(primary, ghost, 3, 1, g=primary) if ghost else nn.Identity()
        self.c2 = c2

    def forward(self, x: Tensor) -> Tensor:
        primary = self.primary(x)
        ghost = self.cheap(primary) if self.c2 - primary else primary[:, :0]
        y = torch.cat([primary, ghost], dim=1)
        b, c, h, w = y.shape
        if c % 2 == 0:
            y = y.reshape(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)
        return y


class GSBottleneck(nn.Module):
    def __init__(self, channels: int, use_sam: bool = True) -> None:
        super().__init__()
        self.conv1 = GSConv2D(channels, channels, 1)
        self.sam = SAM(channels) if use_sam else nn.Identity()
        self.conv2 = GSConv2D(channels, channels, 3)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.sam(self.conv1(x)))


class SPP(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.reduce = ConvBNAct(channels, hidden, 1)
        self.fuse = ConvBNAct(hidden * 4, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        pooled = [x]
        for k in (5, 9, 13):
            pooled.append(F.max_pool2d(x, k, 1, k // 2))
        return self.fuse(torch.cat(pooled, dim=1))


class WeightedFusion(nn.Module):
    def __init__(self, n: int = 2) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n, dtype=torch.float32))

    def forward(self, *xs: Tensor) -> Tensor:
        if len(xs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} tensors, got {len(xs)}")
        w = F.relu(self.weights) + 1e-4
        y = sum(weight * x for weight, x in zip(w, xs)) / w.sum()
        return y


class DCMNet(nn.Module):
    """Compact DCMNet reconstruction producing P3/P4/P5 features."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 16, 3, 2)
        self.stage1 = MobileBottleneck(16, 16, 2, 1)
        self.down2 = MCDownsample(16, 24)
        self.stage2 = nn.Sequential(MobileBottleneck(24, 24, 2), DCM(24))
        self.down3 = MCDownsample(24, 40)
        self.stage3 = nn.Sequential(MobileBottleneck(40, 40, 4), DCM(40))
        self.down4 = MCDownsample(40, 80)
        self.stage4 = nn.Sequential(MobileBottleneck(80, 80, 4), DCM(80), SAM(80))
        self.down5 = MCDownsample(80, 160)
        self.stage5 = nn.Sequential(MobileBottleneck(160, 160, 4), DCM(160), SAM(160))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down2(x))
        p3 = self.stage3(self.down3(x))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))
        return p3, p4, p5


class GMF(nn.Module):
    """Lightweight multi-scale neck with SPP and GS bottleneck fusion."""

    def __init__(self, in_channels: Iterable[int] = (40, 80, 160), channels: int = 128) -> None:
        super().__init__()
        c3, c4, c5 = in_channels
        self.lat3 = ConvBNAct(c3, channels, 1)
        self.lat4 = ConvBNAct(c4, channels, 1)
        self.lat5 = ConvBNAct(c5, channels, 1)
        self.spp = SPP(channels)
        self.fuse_p4 = WeightedFusion(2)
        self.fuse_p3 = WeightedFusion(2)
        self.fuse_p4_down = WeightedFusion(2)
        self.fuse_p5_down = WeightedFusion(2)
        self.gs4 = GSBottleneck(channels)
        self.gs3 = GSBottleneck(channels)
        self.gs4_down = GSBottleneck(channels)
        self.gs5_down = GSBottleneck(channels)
        self.down = ConvBNAct(channels, channels, 3, 2)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = features
        p3 = self.lat3(p3)
        p4 = self.lat4(p4)
        p5 = self.spp(self.lat5(p5))

        p4_td = self.gs4(self.fuse_p4(p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")))
        p3_td = self.gs3(self.fuse_p3(p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")))
        p4_out = self.gs4_down(self.fuse_p4_down(p4_td, self.down(p3_td)))
        p5_out = self.gs5_down(self.fuse_p5_down(p5, self.down(p4_out)))
        return p3_td, p4_out, p5_out


class DecoupledHeadBranch(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(channels, channels, 1)
        self.cls = nn.Sequential(ConvBNAct(channels, channels, 3), ConvBNAct(channels, channels, 3))
        self.reg = nn.Sequential(ConvBNAct(channels, channels, 3), ConvBNAct(channels, channels, 3))
        self.cls_pred = nn.Conv2d(channels, num_classes, 1)
        self.obj_pred = nn.Conv2d(channels, 1, 1)
        self.reg_pred = nn.Conv2d(channels, 4, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        cls = self.cls(x)
        reg = self.reg(x)
        return self.cls_pred(cls), self.obj_pred(reg), self.reg_pred(reg)


class YOLOSM(nn.Module):
    strides = (8, 16, 32)

    def __init__(self, num_classes: int = 1, neck_channels: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = DCMNet()
        self.neck = GMF((40, 80, 160), neck_channels)
        self.heads = nn.ModuleList(
            [DecoupledHeadBranch(neck_channels, num_classes) for _ in self.strides]
        )

    @staticmethod
    def _make_points(h: int, w: int, stride: int, device: torch.device) -> Tensor:
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack([(x + 0.5) * stride, (y + 0.5) * stride], dim=-1).reshape(-1, 2)

    def decode(self, outputs: tuple[list[Tensor], list[Tensor], list[Tensor]]) -> dict[str, Tensor]:
        cls_outputs, obj_outputs, reg_outputs = outputs
        points, strides, cls_logits, obj_logits, reg_logits = [], [], [], [], []
        for cls, obj, reg, stride in zip(cls_outputs, obj_outputs, reg_outputs, self.strides):
            b, _, h, w = cls.shape
            points.append(self._make_points(h, w, stride, cls.device))
            strides.append(torch.full((h * w,), stride, device=cls.device, dtype=torch.float32))
            cls_logits.append(cls.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes))
            obj_logits.append(obj.permute(0, 2, 3, 1).reshape(b, -1, 1))
            reg_logits.append(reg.permute(0, 2, 3, 1).reshape(b, -1, 4))

        points = torch.cat(points, dim=0)
        strides = torch.cat(strides, dim=0)
        cls_logits = torch.cat(cls_logits, dim=1)
        obj_logits = torch.cat(obj_logits, dim=1)
        reg_logits = torch.cat(reg_logits, dim=1)
        distances = reg_logits.clamp(min=-8.0, max=8.0).exp() * strides.view(1, -1, 1)
        centers = points.view(1, -1, 2)
        boxes = torch.cat(
            [centers - distances[..., :2], centers + distances[..., 2:]], dim=-1
        )
        return {
            "cls_logits": cls_logits,
            "obj_logits": obj_logits,
            "reg_logits": reg_logits,
            "boxes": boxes,
            "points": points,
            "strides": strides,
        }

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        features = self.neck(self.backbone(x))
        outputs = tuple(zip(*(head(feature) for head, feature in zip(self.heads, features))))
        decoded = self.decode(outputs)  # type: ignore[arg-type]
        decoded["raw_cls"] = outputs[0]
        decoded["raw_obj"] = outputs[1]
        decoded["raw_reg"] = outputs[2]
        return decoded


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
