"""ANZD-PIDNet v5: pretrained, recipe-first small-defect segmentation.

v5 returns to the PIDNet-S dimensions so the official ImageNet checkpoint
loads, keeps the v3 boundary/signed-distance shape head with v4's
foreground-only, context-verified correction, and drops the randomly
initialized contrast gate, scale fusion, and edge refiner. Inference is one
forward pass; ANZD zoom/component terms remain training-only.
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



class ShapeRefinementHead(nn.Module):
    """Native 1/4-resolution shape refinement with a context verifier.

    The semantic correction is one signed foreground channel rather than an
    unconstrained two-class delta. A verifier gate, conditioned on fused
    features and coarse foreground probability, limits corrections in regions
    where the coarse semantic path sees only background.
    """

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
        if num_classes != 2:
            raise ValueError("ANZD-PIDNet v5 currently supports binary segmentation (num_classes=2).")
        self.context_verifier = nn.Sequential(
            nn.Conv2d(refine_channels + 1, hidden, 1, bias=False),
            BatchNorm2d(hidden, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # A single signed foreground correction has fewer ways to create a
        # background-logit artifact than a free two-class correction map.
        self.semantic_correction = nn.Conv2d(refine_channels, 1, 1, bias=True)

    def forward(self, quarter_features, coarse_quarter):
        detail = self.detail(quarter_features)
        coarse = self.coarse_projection(coarse_quarter)
        fused = self.fuse(torch.cat([detail, coarse], dim=1))
        boundary_logits = self.boundary_head(fused)
        distance_logits = self.distance_head(fused)
        gate = self.shape_gate(torch.cat([fused, boundary_logits, distance_logits], dim=1))
        # Gate the shared refinement features before projecting to the
        gated_features = fused * gate
        coarse_foreground = F.softmax(coarse_quarter.float(), dim=1)[:, 1:2].to(fused.dtype)
        verifier = self.context_verifier(torch.cat([gated_features, coarse_foreground], dim=1))
        correction = self.semantic_correction(gated_features)
        # Keep a small correction floor so genuinely tiny defects that are weak
        # in the coarse map can still be recovered.
        correction = correction * (0.25 + 0.75 * verifier)
        background_delta = torch.zeros_like(correction)
        refined = torch.cat(
            [coarse_quarter[:, :1] + background_delta, coarse_quarter[:, 1:2] + correction],
            dim=1,
        )
        return refined, boundary_logits, distance_logits



class PIDNet(nn.Module):
    """ANZD-PIDNet v5: ImageNet-initializable PIDNet-S with an optional shape head.

    With ``planes=32`` every official PIDNet-S parameter keeps its name and
    shape, so the ImageNet checkpoint loads directly. v3/v4's contrast gate,
    selective scale fusion, and edge refiner are removed: they were randomly
    initialized, sat in front of or beside the pretrained trunk, and showed no
    measurable gain in the v1-v4 runs.
    """

    def __init__(
        self,
        m=2,
        n=3,
        num_classes=2,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
        shape_refine=True,
    ):
        super().__init__()
        self.augment = augment
        self.shape_refine = shape_refine
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
            # With the shape head, boundary supervision comes from its 1/4-res
            # boundary map, so PIDNet's auxiliary D head would be unused.
            if not shape_refine:
                self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)
        if shape_refine:
            self.shape_head = ShapeRefinementHead(planes, num_classes, max(planes, 32))
        self._init_weights()
        if shape_refine:
            # Start from the (pretrained) coarse PIDNet solution and let the
            # shape path learn a correction instead of destabilizing it.
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
        x = self.conv1(x)
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
        final = self.final_layer(self.dfm(p, x, d))
        if not self.shape_refine:
            # Official PIDNet-S outputs: [aux P semantic, final semantic, D boundary].
            if self.augment:
                return [self.seghead_p(auxiliary_p), final, self.seghead_d(auxiliary_d)]
            return final
        coarse_quarter = F.interpolate(
            final, size=quarter_features.shape[-2:], mode="bilinear", align_corners=ALIGN_CORNERS
        )
        final, boundary_logits, distance_logits = self.shape_head(quarter_features, coarse_quarter)
        if self.augment:
            return [self.seghead_p(auxiliary_p), final, boundary_logits, distance_logits]
        return final


# Official ImageNet PIDNet-S checkpoint (XuJiacong/PIDNet README).
PIDNET_S_IMAGENET_GDRIVE_ID = "1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-"
# The ImageNet-pretrained trunk every run must receive; other matched layers
# (P/D branches, PAPPM, fusion) are reported but not required.
PRETRAINED_CORE_PREFIXES = ("conv1.", "layer1.", "layer2.", "layer3.", "layer4.", "layer5.")


def load_imagenet_pretrained(model: nn.Module, checkpoint_path) -> dict:
    """Load an official PIDNet ImageNet checkpoint by name and shape.

    Mirrors the official ``get_seg_model`` filtering, plus ``module.``/``model.``
    prefix stripping. Returns coverage statistics so callers can refuse a run
    whose trunk did not actually load.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        # Older official checkpoints pickle non-tensor metadata alongside the weights.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value

    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in cleaned.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    shape_mismatched = sorted(
        key for key, value in cleaned.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    )
    model_state.update(matched)
    model.load_state_dict(model_state, strict=True)

    def is_core(key):
        return key.startswith(PRETRAINED_CORE_PREFIXES) and "num_batches_tracked" not in key

    core_keys = [key for key in model_state if is_core(key)]
    core_loaded = [key for key in core_keys if key in matched]
    return {
        "checkpoint_tensors": len(cleaned),
        "loaded_tensors": len(matched),
        "model_tensors": len(model_state),
        "core_tensors": len(core_keys),
        "core_loaded": len(core_loaded),
        "core_coverage": len(core_loaded) / max(len(core_keys), 1),
        "total_coverage": len(matched) / max(len(model_state), 1),
        "core_missing": sorted(set(core_keys) - set(core_loaded))[:20],
        "shape_mismatched": shape_mismatched[:20],
    }

def resize_pidnet_outputs(outputs, size):
    """Upsample all PIDNet v5 outputs to a common label resolution."""
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
    """PIDNet loss plus the signed-distance shape objective.

    Outputs are ``[auxiliary_semantic, final_semantic, boundary_logits,
    distance_logits]``.  The distance term is optional for compatibility with
    the original three-output PIDNet (``shape_refine=False``).
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
            raise ValueError("distance_target is required for the four-output PIDNet v5")
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



def build_pidnet_v5(num_classes=2, augment=True, shape_refine=True):
    """PIDNet-S dimensions (planes 32, PAPPM 96, head 128) so ImageNet weights load."""
    return PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=augment,
        shape_refine=shape_refine,
    )
