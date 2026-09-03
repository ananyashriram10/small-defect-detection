"""Fast ANZD-PIDNet architecture, geometry, gradient, and metric verification."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR.parent))

from anzd_pidnet.pidnet import (  # noqa: E402
    BoundaryLoss,
    OhemCrossEntropy,
    build_pidnet_s,
    pidnet_loss,
    resize_pidnet_outputs,
)
from anzd_pidnet.zoom_utils import (  # noqa: E402
    SegmentationMetrics,
    choose_area_normalized_crop,
    component_balanced_recall_loss,
    connected_component_map,
    extract_zoom_arrays,
    generate_boundary,
    quality_gated_zoom_distillation_loss,
)


def verify_geometry():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[50:56, 61:69] = 1
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[..., 1] = 80
    crop = choose_area_normalized_crop(
        mask,
        target_area_ratio=0.08,
        eligible_max_area_ratio=0.01,
        context_scale=1.5,
        minimum_side=16,
        jitter_fraction=0.0,
    )
    assert crop is not None
    assert crop.original_area_ratio < 0.01
    zoom_image, zoom_mask = extract_zoom_arrays(image, mask, crop, 64)
    assert zoom_image.shape == (64, 64, 3)
    assert zoom_mask.shape == (64, 64)
    assert zoom_mask.sum() > mask.sum(), "the small component should occupy more pixels after zooming"
    assert crop.crop_area_ratio > crop.original_area_ratio
    components = connected_component_map(mask)
    assert set(np.unique(components)) == {0, 1}
    boundary = generate_boundary(mask)
    assert boundary.shape == mask.shape and boundary.sum() > 0
    print("geometry: OK", crop)


def verify_model_and_losses():
    torch.manual_seed(7)
    model = build_pidnet_s()
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 7_500_000 < parameter_count < 8_000_000, parameter_count

    images = torch.randn(2, 3, 128, 128)
    labels = torch.zeros(2, 128, 128, dtype=torch.long)
    labels[0, 48:56, 50:58] = 1
    labels[1, 70:82, 31:43] = 1
    boundary = torch.stack(
        [torch.from_numpy(generate_boundary(label.numpy().astype(np.uint8))) for label in labels]
    )
    components = torch.stack(
        [torch.from_numpy(connected_component_map(label.numpy().astype(np.uint8))) for label in labels]
    )
    outputs = resize_pidnet_outputs(model(images), labels.shape[-2:])
    assert [tuple(output.shape) for output in outputs] == [
        (2, 2, 128, 128),
        (2, 2, 128, 128),
        (2, 1, 128, 128),
    ]
    semantic_loss = OhemCrossEntropy(min_kept=256)
    boundary_loss = BoundaryLoss()
    base_loss, _ = pidnet_loss(outputs, labels, boundary, semantic_loss, boundary_loss)
    component_loss, component_count = component_balanced_recall_loss(outputs[1], components)
    assert component_count == 2

    zoom_logits = F.interpolate(outputs[1].detach(), (64, 64), mode="bilinear", align_corners=False)
    zoom_labels = F.interpolate(labels[:, None].float(), (64, 64), mode="nearest")[:, 0].long()
    boxes = torch.tensor([[0, 0, 128, 128], [0, 0, 128, 128]])
    distillation_loss, stats = quality_gated_zoom_distillation_loss(
        outputs[1], zoom_logits, boxes, zoom_labels, minimum_teacher_confidence=0.0
    )
    total = base_loss + component_loss + distillation_loss
    assert torch.isfinite(total)
    total.backward()
    gradient_parameters = sum(
        1
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None and torch.isfinite(parameter.grad).all()
    )
    trainable_tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    assert gradient_parameters > 0.90 * trainable_tensors, (gradient_parameters, trainable_tensors)
    print(
        "model/loss: OK",
        {"parameters": parameter_count, "loss": float(total.detach()), "teacher_stats": stats},
    )


def verify_metrics():
    target = np.zeros((32, 32), dtype=np.uint8)
    target[3:8, 3:8] = 1
    target[20:24, 22:26] = 1
    perfect = SegmentationMetrics()
    perfect.update(target, target)
    result = perfect.finalize()
    for key in ("precision", "recall", "iou", "dice", "component_recall_iou50"):
        assert abs(result[key] - 1.0) < 1e-9, (key, result[key])

    missed = target.copy()
    missed[20:24, 22:26] = 0
    partial = SegmentationMetrics()
    partial.update(missed, target)
    partial_result = partial.finalize()
    assert partial_result["component_recall_iou50"] == 0.5
    assert partial_result["recall"] < 1.0
    print("metrics: OK", partial_result)


def verify_real_sample():
    candidates = list((PROJECT_DIR / "processed_output").glob("*/small/masks/*"))
    candidates = [path for path in candidates if path.is_file() and not path.name.startswith("._")]
    if not candidates:
        print("real sample: SKIPPED (processed_output not present)")
        return
    mask_path = crop = None
    for candidate in candidates:
        mask = Image.open(candidate).convert("L").resize((640, 640), Image.Resampling.NEAREST)
        mask_array = (np.asarray(mask) > 0).astype(np.uint8)
        crop = choose_area_normalized_crop(mask_array, jitter_fraction=0.0)
        if crop is not None:
            mask_path = candidate
            break
    assert crop is not None and mask_path is not None, "no small-bucket mask has an eligible component"
    assert 0 < crop.original_area_ratio <= 0.01
    assert crop.crop_area_ratio > crop.original_area_ratio
    print("real sample: OK", mask_path, crop)


if __name__ == "__main__":
    cv2.setNumThreads(0)
    verify_geometry()
    verify_model_and_losses()
    verify_metrics()
    verify_real_sample()
    print("ALL CHECKS PASSED")
