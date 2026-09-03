"""Faithful, framework-free PIDNet-S model and training losses for ANZD-PIDNet.

The architecture and the three-part PIDNet loss are ported from the project's
existing ``baselines/segmentation/PIDNet/pidnet_s_kaggle.ipynb`` baseline.  The
zoom-distillation experiment deliberately does not change this deployed model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1
ALIGN_CORNERS = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            size = [x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor]
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=ALIGN_CORNERS)
        return out


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(5, stride=2, padding=2),
            BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(9, stride=4, padding=4),
            BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(17, stride=8, padding=8),
            BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False),
        )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, 1, bias=False),
        )
        self.scale_process = nn.Sequential(
            BatchNorm2d(branch_planes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes * 4,
                branch_planes * 4,
                3,
                padding=1,
                groups=4,
                bias=False,
            ),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, 1, bias=False),
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        x0 = self.scale0(x)
        scales = [
            F.interpolate(self.scale1(x), (height, width), mode="bilinear", align_corners=ALIGN_CORNERS) + x0,
            F.interpolate(self.scale2(x), (height, width), mode="bilinear", align_corners=ALIGN_CORNERS) + x0,
            F.interpolate(self.scale3(x), (height, width), mode="bilinear", align_corners=ALIGN_CORNERS) + x0,
            F.interpolate(self.scale4(x), (height, width), mode="bilinear", align_corners=ALIGN_CORNERS) + x0,
        ]
        scale_out = self.scale_process(torch.cat(scales, dim=1))
        return self.compression(torch.cat([x0, scale_out], dim=1)) + self.shortcut(x)


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False), BatchNorm2d(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False), BatchNorm2d(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, 1, bias=False), BatchNorm2d(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.shape[-2:]
        if self.after_relu:
            x, y = self.relu(x), self.relu(y)
        y_q = F.interpolate(self.f_y(y), input_size, mode="bilinear", align_corners=False)
        x_k = self.f_x(x)
        if self.with_channel:
            similarity = torch.sigmoid(self.up(x_k * y_q))
        else:
            similarity = torch.sigmoid(torch.sum(x_k * y_q, dim=1, keepdim=True))
        y = F.interpolate(y, input_size, mode="bilinear", align_corners=False)
        return (1 - similarity) * x + similarity * y


class LightBag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), BatchNorm2d(out_channels)
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), BatchNorm2d(out_channels)
        )

    def forward(self, p, i, d):
        edge_attention = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_attention) * i + p)
        i_add = self.conv_i(i + edge_attention * p)
        return p_add + i_add


class PIDNet(nn.Module):
    """PIDNet-S (`m=2`, `n=3`, `planes=32`) for two-class segmentation."""

    def __init__(
        self,
        m=2,
        n=3,
        num_classes=2,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    ):
        super().__init__()
        self.augment = augment
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
            BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
            BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
        self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
        self.diff3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes, 3, padding=1, bias=False),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.diff4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
            BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
        self.dfm = LightBag(planes * 4, planes * 4)
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        if augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(inplanes, planes, no_relu=index == blocks - 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_single_layer(block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        output_size = (x.shape[-2] // 8, x.shape[-1] // 8)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        p = self.layer3_p(x)
        d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        p = self.pag3(p, self.compression3(x))
        d = d + F.interpolate(self.diff3(x), output_size, mode="bilinear", align_corners=ALIGN_CORNERS)
        if self.augment:
            auxiliary_p = p

        x = self.relu(self.layer4(x))
        p = self.layer4_p(self.relu(p))
        d = self.layer4_d(self.relu(d))
        p = self.pag4(p, self.compression4(x))
        d = d + F.interpolate(self.diff4(x), output_size, mode="bilinear", align_corners=ALIGN_CORNERS)
        if self.augment:
            auxiliary_d = d

        p = self.layer5_p(self.relu(p))
        d = self.layer5_d(self.relu(d))
        x = F.interpolate(
            self.spp(self.layer5(x)), output_size, mode="bilinear", align_corners=ALIGN_CORNERS
        )
        final = self.final_layer(self.dfm(p, x, d))
        if self.augment:
            return [self.seghead_p(auxiliary_p), final, self.seghead_d(auxiliary_d)]
        return final


def resize_pidnet_outputs(outputs, size):
    """Upsample `[P auxiliary, final, D boundary]` outputs to label resolution."""
    return [
        F.interpolate(output, size=size, mode="bilinear", align_corners=ALIGN_CORNERS)
        for output in outputs
    ]


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, threshold=0.9, min_kept=131072, weights=(0.4, 1.0)):
        super().__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="none")

    def plain(self, score, target):
        return self.criterion(score, target).mean()

    def ohem(self, score, target):
        probabilities = F.softmax(score.float(), dim=1)
        pixel_losses = self.criterion(score.float(), target).reshape(-1)
        valid = target.reshape(-1) != self.ignore_label
        safe_target = target.clone()
        safe_target[safe_target == self.ignore_label] = 0
        target_probabilities = probabilities.gather(1, safe_target.unsqueeze(1)).reshape(-1)[valid]
        if target_probabilities.numel() == 0:
            return score.sum() * 0.0
        target_probabilities, order = target_probabilities.sort()
        kth = target_probabilities[min(self.min_kept, target_probabilities.numel() - 1)]
        threshold = max(float(kth.detach()), self.threshold)
        ordered_losses = pixel_losses[valid][order]
        selected = ordered_losses[target_probabilities < threshold]
        if selected.numel() == 0:
            selected = ordered_losses[:1]
        return selected.mean()

    def forward(self, scores, target):
        functions = [self.plain] * (len(self.weights) - 1) + [self.ohem]
        return sum(weight * function(score, target) for weight, score, function in zip(self.weights, scores, functions))


def weighted_boundary_bce(boundary_logits, target):
    logits = boundary_logits.permute(0, 2, 3, 1).reshape(-1)
    target = target.reshape(-1).float()
    positive = target == 1
    negative = target == 0
    positive_count = positive.sum()
    negative_count = negative.sum()
    total = positive_count + negative_count
    if total == 0:
        return logits.sum() * 0.0
    weights = torch.zeros_like(logits)
    weights[positive] = negative_count.float() / total.float()
    weights[negative] = positive_count.float() / total.float()
    return F.binary_cross_entropy_with_logits(logits.float(), target, weights, reduction="mean")


class BoundaryLoss(nn.Module):
    def __init__(self, coefficient=20.0):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, boundary_logits, target):
        return self.coefficient * weighted_boundary_bce(boundary_logits, target)


def pidnet_loss(outputs, labels, boundary_target, semantic_loss, boundary_loss, ignore_label=255):
    """Official PIDNet semantic, boundary, and boundary-aware semantic loss."""
    semantic = semantic_loss(outputs[:-1], labels)
    boundary = boundary_loss(outputs[-1], boundary_target)
    ignored = torch.full_like(labels, ignore_label)
    boundary_labels = torch.where(torch.sigmoid(outputs[-1][:, 0]) > 0.8, labels, ignored)
    boundary_semantic = semantic_loss.ohem(outputs[-2], boundary_labels)
    total = semantic + boundary + boundary_semantic
    return total, {
        "semantic": semantic.detach(),
        "boundary": boundary.detach(),
        "boundary_semantic": boundary_semantic.detach(),
    }


def build_pidnet_s(num_classes=2, augment=True):
    return PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=augment,
    )
