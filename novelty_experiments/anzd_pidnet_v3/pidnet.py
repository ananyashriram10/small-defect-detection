"""ANZD-PIDNet v3: a fast, shape-aware PIDNet segmentation model.

V3 keeps PIDNet's efficient P/I/D streams and the V1 training interface, while
adding three inference-time paths that address the observed errors: a gated
local-contrast stream, explicit 1/4-resolution detail fusion, and a boundary
plus signed-distance shape head.  There is still one model and one forward pass
at inference; no SAM/SAM2 weights or prompts are used.
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


class DepthwiseSeparableBlock(nn.Module):
    """Small spatial block used on the high-resolution detail path."""

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = BatchNorm2d(in_channels, momentum=BN_MOMENTUM)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_bn = BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.depthwise_bn(self.depthwise(x)))
        return self.relu(self.pointwise_bn(self.pointwise(x)))


class LocalContrastGate(nn.Module):
    """Extract inexpensive local image texture and gate it into 1/4 features.

    The branch is intentionally shallow: a stride-4 5x5 projection preserves
    local contrast without creating a second high-resolution backbone.  The
    learned gate can suppress texture-dominated evidence or retain useful
    defect edges on a per-pixel, per-channel basis.
    """

    def __init__(self, feature_channels):
        super().__init__()
        self.texture = nn.Sequential(
            nn.Conv2d(3, feature_channels, 5, stride=4, padding=2, bias=False),
            BatchNorm2d(feature_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                3,
                padding=1,
                groups=feature_channels,
                bias=False,
            ),
            BatchNorm2d(feature_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        hidden = max(feature_channels // 2, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2, hidden, 1, bias=False),
            BatchNorm2d(hidden, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, feature_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, image, quarter_features):
        texture = self.texture(image)
        if texture.shape[-2:] != quarter_features.shape[-2:]:
            texture = F.interpolate(
                texture, quarter_features.shape[-2:], mode="bilinear", align_corners=ALIGN_CORNERS
            )
        gate = self.gate(torch.cat([quarter_features, texture], dim=1))
        return quarter_features + gate * texture


class SelectiveScaleFusion(nn.Module):
    """Fuse 1/4 detail into PIDNet's 1/8 context feature with a learned gate."""

    def __init__(self, context_channels, detail_channels):
        super().__init__()
        self.detail_projection = nn.Sequential(
            nn.Conv2d(detail_channels, context_channels, 1, bias=False),
            BatchNorm2d(context_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        hidden = max(context_channels // 4, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(context_channels * 2, hidden, 1, bias=False),
            BatchNorm2d(hidden, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, context_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(context_channels, context_channels, 3, padding=1, groups=context_channels, bias=False),
            BatchNorm2d(context_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, context, detail):
        projected = self.detail_projection(detail)
        projected = F.interpolate(projected, context.shape[-2:], mode="bilinear", align_corners=ALIGN_CORNERS)
        gate = self.gate(torch.cat([context, projected], dim=1))
        return self.refine(context + gate * projected)


class ShapeRefinementHead(nn.Module):
    """Native 1/4-resolution semantic, boundary, and distance refinement."""

    def __init__(self, detail_channels, num_classes, refine_channels):
        super().__init__()
        self.detail = DepthwiseSeparableBlock(detail_channels, refine_channels)
        self.coarse_projection = nn.Sequential(
            nn.Conv2d(num_classes, refine_channels, 1, bias=False),
            BatchNorm2d(refine_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.fuse = DepthwiseSeparableBlock(refine_channels * 2, refine_channels)
        self.boundary_head = nn.Conv2d(refine_channels, 1, 1, bias=True)
        self.distance_head = nn.Conv2d(refine_channels, 1, 1, bias=True)
        hidden = max(refine_channels // 2, 8)
        self.shape_gate = nn.Sequential(
            nn.Conv2d(refine_channels + 2, hidden, 1, bias=False),
            BatchNorm2d(hidden, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, refine_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.semantic_correction = nn.Conv2d(refine_channels, num_classes, 1, bias=True)

    def forward(self, quarter_features, coarse_quarter):
        detail = self.detail(quarter_features)
        coarse = self.coarse_projection(coarse_quarter)
        fused = self.fuse(torch.cat([detail, coarse], dim=1))
        boundary_logits = self.boundary_head(fused)
        distance_logits = self.distance_head(fused)
        gate = self.shape_gate(torch.cat([fused, boundary_logits, distance_logits], dim=1))
        # Gate the shared refinement features before projecting to the
        # two-class correction map; the gate has ``refine_channels`` while the
        # semantic projection has ``num_classes`` output channels.
        correction = self.semantic_correction(fused * gate)
        return coarse_quarter + correction, boundary_logits, distance_logits


class PIDNet(nn.Module):
    """ANZD-PIDNet v3: widened PIDNet with gated detail and shape refinement."""

    def __init__(
        self,
        m=2,
        n=3,
        num_classes=2,
        planes=40,
        ppm_planes=None,
        head_planes=None,
        augment=True,
        shape_refine=True,
    ):
        super().__init__()
        self.augment = augment
        self.shape_refine = shape_refine
        ppm_planes = ppm_planes if ppm_planes is not None else planes * 3
        head_planes = head_planes if head_planes is not None else planes * 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.local_contrast = LocalContrastGate(planes)
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
        self.scale_fusion = SelectiveScaleFusion(planes * 4, planes)

        if augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            # The shape head replaces PIDNet's auxiliary D segmentation head
            # in v3. Keep the old head only for the explicit shape_refine=False
            # compatibility path so it is not an unused trainable parameter.
            if not shape_refine:
                self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)
        if shape_refine:
            self.shape_head = ShapeRefinementHead(planes, num_classes, max(planes, 32))
        self._init_weights()
        # Start the new texture path conservatively; its gate can open when
        # local contrast is useful but should not overwhelm PIDNet on epoch 1.
        nn.init.constant_(self.local_contrast.gate[-2].bias, -2.0)
        if shape_refine:
            # Start from the stable coarse PIDNet solution and let the shape
            # path learn a correction instead of destabilizing early training.
            nn.init.zeros_(self.shape_head.semantic_correction.weight)
            nn.init.zeros_(self.shape_head.semantic_correction.bias)

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
        image = x
        x = self.conv1(x)
        x = self.local_contrast(image, x)
        quarter_features = x
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
        fused = self.dfm(p, x, d)
        fused = self.scale_fusion(fused, quarter_features)
        final = self.final_layer(fused)
        quarter_size = quarter_features.shape[-2:]
        coarse_quarter = F.interpolate(final, size=quarter_size, mode="bilinear", align_corners=ALIGN_CORNERS)
        if self.shape_refine:
            final, boundary_logits, distance_logits = self.shape_head(quarter_features, coarse_quarter)
        else:
            final = coarse_quarter
            if self.augment:
                boundary_logits = F.interpolate(
                    self.seghead_d(auxiliary_d), size=quarter_size, mode="bilinear", align_corners=ALIGN_CORNERS
                )
            else:
                boundary_logits = torch.zeros(
                    final.shape[0], 1, quarter_size[0], quarter_size[1],
                    device=final.device, dtype=final.dtype,
                )
            distance_logits = torch.zeros_like(boundary_logits)
        if self.augment:
            return [self.seghead_p(auxiliary_p), final, boundary_logits, distance_logits]
        return final


def resize_pidnet_outputs(outputs, size):
    """Upsample all PIDNet v3 outputs to a common label resolution."""
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


def pidnet_loss(
    outputs,
    labels,
    boundary_target,
    semantic_loss,
    boundary_loss,
    distance_target=None,
    distance_weight=0.25,
    ignore_label=255,
):
    """PIDNet loss plus the v3 signed-distance shape objective.

    Outputs are ``[auxiliary_semantic, final_semantic, boundary_logits,
    distance_logits]``.  The distance term is optional for compatibility with
    the original three-output PIDNet, but v3 always supplies its target.
    """
    if len(outputs) < 3:
        raise ValueError(f"Expected at least three PIDNet outputs, got {len(outputs)}")
    semantic_outputs = outputs[:2] if len(outputs) >= 4 else outputs[:-1]
    boundary_logits = outputs[-2] if len(outputs) >= 4 else outputs[-1]
    semantic = semantic_loss(semantic_outputs, labels)
    boundary = boundary_loss(boundary_logits, boundary_target)
    ignored = torch.full_like(labels, ignore_label)
    boundary_labels = torch.where(torch.sigmoid(boundary_logits[:, 0]) > 0.8, labels, ignored)
    boundary_semantic = semantic_loss.ohem(semantic_outputs[-1], boundary_labels)
    total = semantic + boundary + boundary_semantic
    distance = total.sum() * 0.0
    if len(outputs) >= 4:
        if distance_target is None:
            raise ValueError("distance_target is required for the four-output PIDNet v3")
        predicted_distance = torch.tanh(outputs[-1][:, 0].float())
        valid = (labels != ignore_label).float()
        distance_error = F.smooth_l1_loss(predicted_distance, distance_target.float(), reduction="none")
        distance = (distance_error * valid).sum() / valid.sum().clamp_min(1.0)
        total = total + distance_weight * distance
    parts = {
        "semantic": semantic.detach(),
        "boundary": boundary.detach(),
        "boundary_semantic": boundary_semantic.detach(),
    }
    if len(outputs) >= 4:
        parts["distance"] = distance.detach()
    return total, parts


def build_pidnet_v3(num_classes=2, augment=True, planes=40, shape_refine=True):
    return PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=planes,
        ppm_planes=planes * 3,
        head_planes=planes * 4,
        augment=augment,
        shape_refine=shape_refine,
    )


def build_pidnet_s(num_classes=2, augment=True):
    """Compatibility alias for the v3 default model."""
    return build_pidnet_v3(num_classes=num_classes, augment=augment)
