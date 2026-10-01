"""Fast ANZD-PIDNet v5 checks: pretrained loading, architecture, losses, metrics, augmentation.

Run before every RunPod launch:
    python novelty_experiments/anzd_pidnet_v5/verify.py

If the official PIDNet-S ImageNet checkpoint is present (PRETRAINED_PATH, or
pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar in the repository), its
trunk coverage is checked too.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install v5 runtime dependencies first with "
        "python -m pip install -r novelty_experiments/anzd_pidnet_v5/requirements.txt "
        "(the RunPod PyTorch image should already provide torch)."
    ) from exc

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR.parent))

from anzd_pidnet_v5.pidnet import (  # noqa: E402
    BoundaryLoss,
    OhemCrossEntropy,
    build_pidnet_v5,
    load_imagenet_pretrained,
    pidnet_loss,
    resize_pidnet_outputs,
)
from anzd_pidnet_v5.zoom_utils import (  # noqa: E402
    SegmentationMetrics,
    augment_pair,
    choose_area_normalized_crop,
    component_area_statistics,
    component_balanced_recall_loss,
    connected_component_map,
    extract_zoom_arrays,
    generate_boundary,
    generate_signed_distance,
    quality_gated_zoom_distillation_loss,
    remove_small_components,
)

PIDNET_S_PARAMETERS = 7_716_549


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
    zoom_image, zoom_mask = extract_zoom_arrays(image, mask, crop, 64)
    assert zoom_image.shape == (64, 64, 3) and zoom_mask.shape == (64, 64)
    assert zoom_mask.sum() > mask.sum(), "the small component should occupy more pixels after zooming"
    print("geometry: OK")


def training_targets(labels):
    arrays = [label.numpy().astype(np.uint8) for label in labels]
    boundary = torch.stack([torch.from_numpy(generate_boundary(array)) for array in arrays])
    distance = torch.stack([torch.from_numpy(generate_signed_distance(array)) for array in arrays])
    components = torch.stack([torch.from_numpy(connected_component_map(array)) for array in arrays])
    return boundary, distance, components


def verify_model_and_losses():
    for shape_refine in (False, True):
        torch.manual_seed(7)
        model = build_pidnet_v5(shape_refine=shape_refine)
        model.train()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if not shape_refine:
            assert parameter_count == PIDNET_S_PARAMETERS, (
                f"shape_refine=False must be exactly PIDNet-S ({PIDNET_S_PARAMETERS:,}), got {parameter_count:,}"
            )
        else:
            assert 7_000_000 < parameter_count < 9_000_000, parameter_count

        images = torch.randn(2, 3, 128, 128)
        labels = torch.zeros(2, 128, 128, dtype=torch.long)
        labels[0, 48:56, 50:58] = 1
        labels[1, 70:82, 31:43] = 1
        boundary, distance, components = training_targets(labels)
        outputs = resize_pidnet_outputs(model(images), labels.shape[-2:])
        expected = [(2, 2, 128, 128), (2, 2, 128, 128), (2, 1, 128, 128)]
        if shape_refine:
            expected.append((2, 1, 128, 128))
        assert [tuple(output.shape) for output in outputs] == expected

        base_loss, parts = pidnet_loss(
            outputs,
            labels,
            boundary,
            OhemCrossEntropy(min_kept=256),
            BoundaryLoss(),
            distance_target=distance,
        )
        assert ("distance" in parts) == shape_refine
        component_loss, component_count = component_balanced_recall_loss(outputs[1], components)
        assert component_count == 2
        zoom_logits = F.interpolate(outputs[1].detach(), (64, 64), mode="bilinear", align_corners=False)
        zoom_labels = F.interpolate(labels[:, None].float(), (64, 64), mode="nearest")[:, 0].long()
        boxes = torch.tensor([[0, 0, 128, 128], [0, 0, 128, 128]])
        distillation_loss, _ = quality_gated_zoom_distillation_loss(
            outputs[1], zoom_logits, boxes, zoom_labels, minimum_teacher_confidence=0.0
        )
        total = base_loss + component_loss + distillation_loss
        assert torch.isfinite(total)
        total.backward()
        with_gradient = sum(
            1 for parameter in model.parameters() if parameter.grad is not None and torch.isfinite(parameter.grad).all()
        )
        trainable = sum(1 for parameter in model.parameters() if parameter.requires_grad)
        assert with_gradient > 0.90 * trainable, (with_gradient, trainable)

        model.eval()
        with torch.no_grad():
            deployed = build_pidnet_v5(augment=False, shape_refine=shape_refine)
            deployed.load_state_dict(
                {key: value for key, value in model.state_dict().items() if key in deployed.state_dict()}
            )
            deployed.eval()
            assert torch.allclose(deployed(images), model(images)[1], atol=1e-5), "deployed path diverges"
        print(f"model/loss (shape_refine={shape_refine}): OK", {"parameters": parameter_count})


def verify_pretrained_loading():
    torch.manual_seed(1)
    donor = build_pidnet_v5(shape_refine=False)
    donor_state = {f"module.{key}": value for key, value in donor.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "donor.pth.tar"
        torch.save({"state_dict": donor_state, "epoch": 1}, path)

        torch.manual_seed(2)
        receiver = build_pidnet_v5(shape_refine=True)
        report = load_imagenet_pretrained(receiver, path)
        assert report["core_coverage"] == 1.0, report
        receiver_state = receiver.state_dict()
        for key in ("conv1.0.weight", "layer3.2.conv2.weight", "layer5.1.bn3.running_mean", "spp.scale0.2.weight"):
            assert torch.equal(receiver_state[key], donor.state_dict()[key]), key
        # The new head must keep its zero-initialized correction.
        assert receiver_state["shape_head.semantic_correction.weight"].abs().sum() == 0

        # A width-mismatched checkpoint must be detected rather than silently skipped.
        wide = {key: torch.zeros(value.shape[0] + 1, *value.shape[1:]) if value.ndim else value
                for key, value in donor.state_dict().items()}
        torch.save({"state_dict": wide}, path)
        mismatch_report = load_imagenet_pretrained(build_pidnet_v5(shape_refine=True), path)
        assert mismatch_report["core_coverage"] < 0.95, mismatch_report
    print("pretrained loader: OK", {k: report[k] for k in ("loaded_tensors", "core_tensors", "core_coverage")})

    candidates = [
        Path(os.environ["PRETRAINED_PATH"]) if os.environ.get("PRETRAINED_PATH") else None,
        PROJECT_DIR / "pretrained_models" / "imagenet" / "PIDNet_S_ImageNet.pth.tar",
        Path("/workspace/pretrained/PIDNet_S_ImageNet.pth.tar"),
    ]
    official = next((path for path in candidates if path is not None and path.exists()), None)
    if official is None:
        print("official ImageNet checkpoint: SKIPPED (not downloaded here; the trainer downloads and checks it)")
        return
    official_report = load_imagenet_pretrained(build_pidnet_v5(shape_refine=True), official)
    print("official ImageNet checkpoint:", official, official_report)
    assert official_report["core_coverage"] >= 0.95, official_report


def verify_metrics_and_postprocess():
    target = np.zeros((32, 32), dtype=np.uint8)
    target[3:8, 3:8] = 1
    target[20:24, 22:26] = 1
    perfect = SegmentationMetrics()
    perfect.update(target, target)
    for key, value in perfect.finalize().items():
        if key in ("precision", "recall", "iou", "dice", "component_recall_iou50"):
            assert abs(value - 1.0) < 1e-9, (key, value)

    prediction = target.copy().astype(bool)
    prediction[20:24, 22:26] = False
    prediction[20:22, 22:24] = True  # 4-px true positive blob
    prediction[28, 28] = True  # 1-px false positive
    prediction[10:13, 25:28] = True  # 9-px false positive
    areas, blob_tp = component_area_statistics(prediction, target)
    for minimum in (0, 2, 5, 10, 30):
        filtered = remove_small_components(prediction, minimum)
        dropped = areas < minimum
        expected_tp = int(np.logical_and(prediction, target).sum()) - int(blob_tp[dropped].sum())
        assert int(np.logical_and(filtered, target).sum()) == expected_tp, minimum
        expected_fp = int((prediction & ~target.astype(bool)).sum()) - int((areas[dropped] - blob_tp[dropped]).sum())
        assert int((filtered & ~target.astype(bool)).sum()) == expected_fp, minimum
    assert remove_small_components(prediction, 5).sum() == prediction.sum() - 5
    print("metrics/post-process: OK")


def verify_augmentation():
    random.seed(3)
    size = 96
    for _ in range(40):
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        rgb[30:50, 40:70] = 255
        mask[30:50, 40:70] = 1
        out_rgb, out_mask = augment_pair(rgb, mask, scale_min=0.75, scale_max=1.5, color_jitter=0.0)
        assert out_rgb.shape == (size, size, 3) and out_mask.shape == (size, size)
        assert set(np.unique(out_mask)) <= {0, 1}
        bright = out_rgb[..., 0] > 127
        # Image and mask must be transformed together (only interpolation edges may differ).
        disagreement = np.logical_xor(bright, out_mask.astype(bool)).sum()
        assert disagreement <= 0.15 * max(out_mask.sum(), 1) + 4 * size, disagreement
    print("augmentation: OK")


def verify_real_sample():
    candidates = [
        path
        for path in (PROJECT_DIR / "processed_output").glob("*/small/masks/*")
        if path.is_file() and not path.name.startswith("._")
    ]
    if not candidates:
        print("real sample: SKIPPED (processed_output not present)")
        return
    for candidate in candidates:
        mask = Image.open(candidate).convert("L").resize((640, 640), Image.Resampling.NEAREST)
        crop = choose_area_normalized_crop((np.asarray(mask) > 0).astype(np.uint8), jitter_fraction=0.0)
        if crop is not None:
            assert 0 < crop.original_area_ratio <= 0.01
            assert crop.crop_area_ratio > crop.original_area_ratio
            print("real sample: OK", candidate.name)
            return
    raise AssertionError("no small-bucket mask has an eligible component")


if __name__ == "__main__":
    cv2.setNumThreads(0)
    verify_geometry()
    verify_model_and_losses()
    verify_pretrained_loading()
    verify_metrics_and_postprocess()
    verify_augmentation()
    verify_real_sample()
    print("ALL CHECKS PASSED")
