"""Area-normalized zoom, distillation, component loss, and metric utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class ZoomCrop:
    box_xyxy: tuple[int, int, int, int]
    component_id: int
    component_area: int
    original_area_ratio: float
    crop_area_ratio: float


def connected_component_map(binary_mask: np.ndarray, min_pixels: int = 1) -> np.ndarray:
    """Return consecutive component ids, with zero reserved for background."""
    mask = np.ascontiguousarray(binary_mask.astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if min_pixels <= 1:
        return labels.astype(np.int32, copy=False)
    output = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= min_pixels:
            output[labels == component_id] = next_id
            next_id += 1
    return output


def generate_boundary(
    binary_mask: np.ndarray,
    edge_size: int = 4,
    y_kernel_size: int = 6,
    x_kernel_size: int = 6,
) -> np.ndarray:
    """Generate PIDNet's Canny-and-dilate boundary supervision."""
    label = np.ascontiguousarray(binary_mask.astype(np.uint8))
    edge = cv2.Canny(label, 0.1, 0.2)
    kernel = np.ones((edge_size, edge_size), np.uint8)
    # Preserve the official PIDNet preprocessing: suppress image-frame edges
    # before dilation, then restore the original canvas with zero padding.
    edge = edge[y_kernel_size:-y_kernel_size, x_kernel_size:-x_kernel_size]
    edge = np.pad(
        edge,
        ((y_kernel_size, y_kernel_size), (x_kernel_size, x_kernel_size)),
        mode="constant",
    )
    return (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)


def choose_area_normalized_crop(
    binary_mask: np.ndarray,
    *,
    target_area_ratio: float = 0.08,
    eligible_max_area_ratio: float = 0.01,
    context_scale: float = 1.5,
    minimum_side: int = 24,
    jitter_fraction: float = 0.08,
    rng: random.Random | None = None,
) -> ZoomCrop | None:
    """Choose one small component and a square crop that magnifies it predictably.

    Components are sampled with inverse-square-root area weighting, so the smallest
    component is favored without deterministically showing the same one every epoch.
    The crop side is chosen so the component would occupy ``target_area_ratio`` of
    the crop, subject to fitting its full bounding box plus context.
    """
    if not 0 < target_area_ratio < 1:
        raise ValueError("target_area_ratio must be between zero and one")
    if not 0 < eligible_max_area_ratio <= 1:
        raise ValueError("eligible_max_area_ratio must be between zero and one")
    rng = rng or random
    mask = np.ascontiguousarray(binary_mask.astype(np.uint8))
    height, width = mask.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = height * width

    eligible = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area > 0 and area / image_area <= eligible_max_area_ratio:
            eligible.append(component_id)
    if not eligible:
        return None

    weights = [1.0 / math.sqrt(float(stats[index, cv2.CC_STAT_AREA])) for index in eligible]
    component_id = rng.choices(eligible, weights=weights, k=1)[0]
    area = int(stats[component_id, cv2.CC_STAT_AREA])
    component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
    component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
    center_x, center_y = (float(value) for value in centroids[component_id])

    ratio_side = math.sqrt(area / target_area_ratio)
    extent_side = max(component_width, component_height) * context_scale
    side = int(math.ceil(max(ratio_side, extent_side, minimum_side)))
    side = min(side, height, width)

    if jitter_fraction > 0:
        center_x += rng.uniform(-jitter_fraction, jitter_fraction) * side
        center_y += rng.uniform(-jitter_fraction, jitter_fraction) * side
    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x1 = min(max(x1, 0), width - side)
    y1 = min(max(y1, 0), height - side)
    x2, y2 = x1 + side, y1 + side
    crop_area_ratio = float((labels[y1:y2, x1:x2] == component_id).sum()) / max(side * side, 1)
    return ZoomCrop(
        box_xyxy=(x1, y1, x2, y2),
        component_id=component_id,
        component_area=area,
        original_area_ratio=area / image_area,
        crop_area_ratio=crop_area_ratio,
    )


def extract_zoom_arrays(
    rgb_image: np.ndarray,
    binary_mask: np.ndarray,
    crop: ZoomCrop,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an integer-aligned crop and resize image/mask with correct kernels."""
    x1, y1, x2, y2 = crop.box_xyxy
    rgb = Image.fromarray(rgb_image[y1:y2, x1:x2]).resize(
        (output_size, output_size), Image.Resampling.BILINEAR
    )
    mask = Image.fromarray((binary_mask[y1:y2, x1:x2] * 255).astype(np.uint8)).resize(
        (output_size, output_size), Image.Resampling.NEAREST
    )
    return np.asarray(rgb).copy(), (np.asarray(mask) > 0).astype(np.uint8)


def crop_logits_to_boxes(
    full_logits: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Differentiably crop each sample's full-resolution logits to its zoom box."""
    crops = []
    height, width = full_logits.shape[-2:]
    for batch_index, box in enumerate(boxes_xyxy.detach().cpu().tolist()):
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, x2 = min(max(x1, 0), width - 1), min(max(x2, 1), width)
        y1, y2 = min(max(y1, 0), height - 1), min(max(y2, 1), height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid zoom box after clipping: {(x1, y1, x2, y2)}")
        crop = full_logits[batch_index : batch_index + 1, :, y1:y2, x1:x2]
        crops.append(F.interpolate(crop, size=output_size, mode="bilinear", align_corners=False))
    return torch.cat(crops, dim=0)


def quality_gated_zoom_distillation_loss(
    full_logits: torch.Tensor,
    zoom_logits: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    zoom_labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    minimum_teacher_confidence: float = 0.55,
    foreground_weight: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Distill trustworthy zoom predictions into the corresponding full-image region.

    The zoom prediction is the detached teacher. Pixels are used only when its class
    agrees with ground truth and confidence is adequate. This prevents an early weak
    self-teacher from reinforcing its own errors.
    """
    if full_logits.shape[0] != zoom_logits.shape[0] or full_logits.shape[0] != boxes_xyxy.shape[0]:
        raise ValueError("full_logits, zoom_logits, and boxes_xyxy must have equal batch size")
    full_crops = crop_logits_to_boxes(full_logits, boxes_xyxy, zoom_logits.shape[-2:]).float()
    teacher_probabilities = F.softmax(zoom_logits.detach().float() / temperature, dim=1)
    teacher_confidence, teacher_class = teacher_probabilities.max(dim=1)
    valid = (teacher_class == zoom_labels) & (teacher_confidence >= minimum_teacher_confidence)
    if not bool(valid.any()):
        return full_logits.sum() * 0.0, {"valid_fraction": 0.0, "foreground_valid_fraction": 0.0}

    per_pixel = F.kl_div(
        F.log_softmax(full_crops / temperature, dim=1),
        teacher_probabilities,
        reduction="none",
    ).sum(dim=1) * (temperature**2)
    weights = torch.ones_like(per_pixel)
    weights = weights + (foreground_weight - 1.0) * (zoom_labels == 1).float()
    weighted_valid = weights * valid.float()
    loss = (per_pixel * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
    foreground = zoom_labels == 1
    foreground_valid = (valid & foreground).sum().float() / foreground.sum().clamp_min(1).float()
    return loss, {
        "valid_fraction": float(valid.float().mean().detach()),
        "foreground_valid_fraction": float(foreground_valid.detach()),
    }


def component_balanced_recall_loss(
    final_logits: torch.Tensor,
    component_maps: torch.Tensor,
    *,
    maximum_area_ratio: float = 0.01,
    minimum_pixels: int = 2,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Give every small connected component one equal recall-oriented loss term.

    PIDNet's original semantic/boundary losses continue to control false positives;
    this auxiliary term prevents large masks from dominating the foreground gradient.
    """
    defect_probability = F.softmax(final_logits.float(), dim=1)[:, 1]
    height, width = defect_probability.shape[-2:]
    losses = []
    for batch_index in range(defect_probability.shape[0]):
        component_map = component_maps[batch_index]
        for component_id in torch.unique(component_map):
            component_id_int = int(component_id.detach())
            if component_id_int == 0:
                continue
            pixels = component_map == component_id_int
            area = int(pixels.sum().detach())
            if area < minimum_pixels or area / (height * width) > maximum_area_ratio:
                continue
            mean_probability = defect_probability[batch_index][pixels].mean()
            losses.append((1.0 - mean_probability).clamp_min(0).pow(gamma))
    if not losses:
        return final_logits.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def _filtered_components(binary_mask: np.ndarray, minimum_pixels: int):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(binary_mask.astype(np.uint8)), connectivity=8
    )
    keep = [index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels]
    remap = np.zeros(count, dtype=np.int32)
    for new_index, old_index in enumerate(keep, start=1):
        remap[old_index] = new_index
    mapped = remap[labels]
    areas = np.array([int(stats[index, cv2.CC_STAT_AREA]) for index in keep], dtype=np.int64)
    return mapped, areas


def _component_candidates(prediction: np.ndarray, target: np.ndarray, minimum_pred_pixels: int):
    pred_labels, pred_areas = _filtered_components(prediction, minimum_pred_pixels)
    target_labels, target_areas = _filtered_components(target, 1)
    pred_count, target_count = len(pred_areas), len(target_areas)
    if pred_count == 0 or target_count == 0:
        return pred_count, target_count, [], np.zeros(target_count, dtype=np.float64)

    encoded = target_labels.astype(np.int64) * (pred_count + 1) + pred_labels.astype(np.int64)
    overlap = np.bincount(encoded.reshape(-1), minlength=(target_count + 1) * (pred_count + 1))
    overlap = overlap.reshape(target_count + 1, pred_count + 1)[1:, 1:]
    candidates = []
    best_per_target = np.zeros(target_count, dtype=np.float64)
    target_indices, pred_indices = np.nonzero(overlap)
    for target_index, pred_index in zip(target_indices.tolist(), pred_indices.tolist()):
        intersection = int(overlap[target_index, pred_index])
        union = int(target_areas[target_index] + pred_areas[pred_index] - intersection)
        iou = intersection / max(union, 1)
        candidates.append((iou, target_index, pred_index))
        best_per_target[target_index] = max(best_per_target[target_index], iou)
    candidates.sort(reverse=True)
    return pred_count, target_count, candidates, best_per_target


def _greedy_match_count(candidates, threshold: float) -> int:
    matched_targets, matched_predictions = set(), set()
    for iou, target_index, pred_index in candidates:
        if iou < threshold:
            break
        if target_index in matched_targets or pred_index in matched_predictions:
            continue
        matched_targets.add(target_index)
        matched_predictions.add(pred_index)
    return len(matched_targets)


class SegmentationMetrics:
    """Pixel and connected-component metrics accumulated without storing predictions."""

    def __init__(self, minimum_pred_component_pixels: int = 3):
        self.minimum_pred_component_pixels = minimum_pred_component_pixels
        self.tp = self.fp = self.fn = self.tn = 0
        self.images = 0
        self.gt_components = self.pred_components = 0
        self.component_tp_iou10 = self.component_tp_iou50 = 0
        self.best_iou_sum = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray):
        prediction = prediction.astype(bool)
        target = target.astype(bool)
        self.tp += int(np.logical_and(prediction, target).sum())
        self.fp += int(np.logical_and(prediction, ~target).sum())
        self.fn += int(np.logical_and(~prediction, target).sum())
        self.tn += int(np.logical_and(~prediction, ~target).sum())
        self.images += 1

        pred_count, target_count, candidates, best_iou = _component_candidates(
            prediction, target, self.minimum_pred_component_pixels
        )
        self.pred_components += pred_count
        self.gt_components += target_count
        self.component_tp_iou10 += _greedy_match_count(candidates, 0.10)
        self.component_tp_iou50 += _greedy_match_count(candidates, 0.50)
        self.best_iou_sum += float(best_iou.sum())

    def merge(self, other: "SegmentationMetrics"):
        for name in (
            "tp",
            "fp",
            "fn",
            "tn",
            "images",
            "gt_components",
            "pred_components",
            "component_tp_iou10",
            "component_tp_iou50",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.best_iou_sum += other.best_iou_sum

    def finalize(self) -> dict[str, float | int]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        iou = self.tp / max(self.tp + self.fp + self.fn, 1)
        dice = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)
        specificity = self.tn / max(self.tn + self.fp, 1)
        accuracy = (self.tp + self.tn) / max(self.tp + self.fp + self.fn + self.tn, 1)
        component_precision10 = self.component_tp_iou10 / max(self.pred_components, 1)
        component_recall10 = self.component_tp_iou10 / max(self.gt_components, 1)
        component_precision50 = self.component_tp_iou50 / max(self.pred_components, 1)
        component_recall50 = self.component_tp_iou50 / max(self.gt_components, 1)

        def harmonic(precision_value, recall_value):
            return 2 * precision_value * recall_value / max(precision_value + recall_value, 1e-12)

        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "dice": dice,
            "pixel_f1": dice,
            "accuracy": accuracy,
            "specificity": specificity,
            "fp_per_image": self.fp / max(self.images, 1),
            "component_precision_iou10": component_precision10,
            "component_recall_iou10": component_recall10,
            "component_f1_iou10": harmonic(component_precision10, component_recall10),
            "component_precision_iou50": component_precision50,
            "component_recall_iou50": component_recall50,
            "component_f1_iou50": harmonic(component_precision50, component_recall50),
            "mean_best_component_iou": self.best_iou_sum / max(self.gt_components, 1),
            "false_positive_components_per_image_iou10": (
                self.pred_components - self.component_tp_iou10
            )
            / max(self.images, 1),
            "images": self.images,
            "gt_components": self.gt_components,
            "pred_components": self.pred_components,
            "true_positive_pixels": self.tp,
            "false_positive_pixels": self.fp,
            "false_negative_pixels": self.fn,
            "true_negative_pixels": self.tn,
        }
