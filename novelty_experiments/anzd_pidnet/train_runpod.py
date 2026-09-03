"""Train ANZD-PIDNet: segmentation-only area-normalized zoom distillation.

The deployed network is the exact PIDNet-S baseline. During training only, small
connected defect components are magnified into fixed-size crops. The shared model
learns those crops with the original PIDNet loss, then confident/correct zoom
predictions supervise the corresponding full-image logits. An optional component-
balanced recall loss gives every small component an equal foreground term.

Modes (set METHOD):
    baseline       exact PIDNet-S control
    component      PIDNet-S + component-balanced recall loss
    zoom           PIDNet-S + area-normalized crop learning/distillation
    zoom_component full proposed method (default)

RunPod example:
    export WANDB_API_KEY=<key>
    export DATASET_ROOT=/workspace/dataset
    export METHOD=zoom_component
    nohup python -u novelty_experiments/anzd_pidnet/train_runpod.py > anzd_pidnet.log 2>&1 &
    tail -f anzd_pidnet.log
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


METHOD = os.environ.get("METHOD", "zoom_component").strip().lower()
VALID_METHODS = {"baseline", "component", "zoom", "zoom_component"}
if METHOD not in VALID_METHODS:
    raise ValueError(f"METHOD must be one of {sorted(VALID_METHODS)}, got {METHOD!r}")
USE_ZOOM = METHOD in {"zoom", "zoom_component"}
USE_COMPONENT_LOSS = METHOD in {"component", "zoom_component"}

MODEL_LABEL = f"ANZD-PIDNet ({METHOD})"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "smallDefectDetection")

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "Smalldefect/SmallDefectDataseet")

SEED = env_int("SEED", 42)
IMG_SIZE = env_int("IMG_SIZE", 640)
CROP_SIZE = env_int("CROP_SIZE", 320)
BATCH_SIZE = env_int("BATCH_SIZE", 8)
MAX_EPOCHS = env_int("MAX_EPOCHS", 50)
PATIENCE = env_int("PATIENCE", 15)
LEARNING_RATE = env_float("LEARNING_RATE", 6e-5)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 1e-2)
NUM_WORKERS = env_int("NUM_WORKERS", 2)
RUN_NAME = os.environ.get("RUN_NAME", f"ANZD_PIDNet_{METHOD}_imgsz{IMG_SIZE}")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", RUN_NAME)

TARGET_CROP_AREA_RATIO = env_float("TARGET_CROP_AREA_RATIO", 0.08)
SMALL_COMPONENT_MAX_AREA_RATIO = env_float("SMALL_COMPONENT_MAX_AREA_RATIO", 0.01)
CROP_CONTEXT_SCALE = env_float("CROP_CONTEXT_SCALE", 1.5)
CROP_MINIMUM_SIDE = env_int("CROP_MINIMUM_SIDE", 24)
CROP_JITTER_FRACTION = env_float("CROP_JITTER_FRACTION", 0.08)
MAX_ZOOM_CROPS_PER_BATCH = env_int("MAX_ZOOM_CROPS_PER_BATCH", 4)

CROP_SUPERVISION_WEIGHT = env_float("CROP_SUPERVISION_WEIGHT", 0.5)
DISTILLATION_WEIGHT = env_float("DISTILLATION_WEIGHT", 1.0)
COMPONENT_LOSS_WEIGHT = env_float("COMPONENT_LOSS_WEIGHT", 0.5)
DISTILLATION_TEMPERATURE = env_float("DISTILLATION_TEMPERATURE", 2.0)
TEACHER_CONFIDENCE = env_float("TEACHER_CONFIDENCE", 0.55)
FOREGROUND_DISTILLATION_WEIGHT = env_float("FOREGROUND_DISTILLATION_WEIGHT", 4.0)
DISTILLATION_START_EPOCH = env_int("DISTILLATION_START_EPOCH", 5)
DISTILLATION_RAMP_EPOCHS = env_int("DISTILLATION_RAMP_EPOCHS", 5)

MIN_PRED_COMPONENT_PIXELS = env_int("MIN_PRED_COMPONENT_PIXELS", 3)
PROGRESS_EVERY = env_int("PROGRESS_EVERY", 50)
ALLOW_CPU = env_bool("ALLOW_CPU", False)
SMOKE_TEST = env_bool("SMOKE_TEST", False)
INSTALL_DEPENDENCIES = env_bool("INSTALL_DEPENDENCIES", True)

if IMG_SIZE % 8 or CROP_SIZE % 8:
    raise ValueError("IMG_SIZE and CROP_SIZE must both be divisible by 8 for PIDNet feature alignment.")
if BATCH_SIZE < 2 and not SMOKE_TEST:
    raise ValueError("Training BATCH_SIZE must be at least 2 because PIDNet's pooled branch uses BatchNorm.")

METHOD_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/anzd_pidnet_runs"))
RUN_DIR = BASE_DIR / "runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

if INSTALL_DEPENDENCIES:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "-r",
            str(METHOD_DIR / "requirements.txt"),
        ],
        check=True,
    )

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(METHOD_DIR.parent))
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


if SMOKE_TEST:
    MAX_EPOCHS = min(MAX_EPOCHS, 1)
    PATIENCE = 1
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    # Exercise the distillation code path during the single smoke epoch.
    DISTILLATION_START_EPOCH = 1
    DISTILLATION_RAMP_EPOCHS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda" and not ALLOW_CPU:
    raise RuntimeError("CUDA is unavailable. Attach a GPU, or set ALLOW_CPU=1 only for a slow smoke test.")
if DEVICE.type == "cuda" and torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected one visible GPU after CUDA_VISIBLE_DEVICES=0, found {torch.cuda.device_count()}.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

print("torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print({"run": RUN_NAME, "method": METHOD, "zoom": USE_ZOOM, "component_loss": USE_COMPONENT_LOSS})


def initialize_wandb():
    mode = os.environ.get("WANDB_MODE", "online")
    if mode not in {"disabled", "offline"}:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()
    return wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        mode=mode,
        config={
            "experiment": RUN_NAME,
            "model": MODEL_LABEL,
            "method": METHOD,
            "image_size": IMG_SIZE,
            "crop_size": CROP_SIZE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "target_crop_area_ratio": TARGET_CROP_AREA_RATIO,
            "small_component_max_area_ratio": SMALL_COMPONENT_MAX_AREA_RATIO,
            "crop_context_scale": CROP_CONTEXT_SCALE,
            "crop_supervision_weight": CROP_SUPERVISION_WEIGHT,
            "distillation_weight": DISTILLATION_WEIGHT,
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "distillation_temperature": DISTILLATION_TEMPERATURE,
            "teacher_confidence": TEACHER_CONFIDENCE,
            "distillation_start_epoch": DISTILLATION_START_EPOCH,
            "distillation_ramp_epochs": DISTILLATION_RAMP_EPOCHS,
            "deployed_architecture_changed": False,
        },
    )


wandb_run = initialize_wandb()


def missing_dataset_buckets():
    return [
        f"{dataset}/{size}"
        for dataset in DATASET_NAMES
        for size in SIZE_BUCKETS
        if not (DATASET_ROOT / dataset / size / "images").exists()
        or not (DATASET_ROOT / dataset / size / "masks").exists()
        or not (DATASET_ROOT / dataset / size / "labels_yolo").exists()
    ]


missing_buckets = missing_dataset_buckets()
if missing_buckets:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"Dataset is incomplete at {DATASET_ROOT}; {len(missing_buckets)} bucket(s) are missing. "
            "Set HF_TOKEN so the private dataset can be downloaded, or populate DATASET_ROOT first."
        )
    from huggingface_hub import snapshot_download

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
    for attempt in range(1, 7):
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir=str(DATASET_ROOT),
                token=token,
                max_workers=4,
            )
            break
        except Exception:
            if attempt == 6:
                raise
            wait_seconds = 30 * attempt
            print(f"Dataset download attempt {attempt} failed; retrying in {wait_seconds}s.")
            time.sleep(wait_seconds)
if missing_dataset_buckets():
    raise RuntimeError(f"Dataset remains incomplete at {DATASET_ROOT} after setup.")


def index_files(directory: Path, suffixes: set[str]):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("._")
    }


def candidate_stems(image_stem: str, target: str):
    base = image_stem.removesuffix("_defect")
    if target == "mask":
        return [image_stem, image_stem.replace("_defect", "_mask"), base, base + "_mask", base + "_gt"]
    return [image_stem, image_stem.replace("_defect", "_bbs"), base, base + "_bbs"]


samples = []
missing_files = 0
for dataset_name in DATASET_NAMES:
    for size_bucket in SIZE_BUCKETS:
        bucket_root = DATASET_ROOT / dataset_name / size_bucket
        image_dir, mask_dir, label_dir = (
            bucket_root / "images",
            bucket_root / "masks",
            bucket_root / "labels_yolo",
        )
        mask_index = index_files(mask_dir, IMAGE_EXTENSIONS)
        label_index = index_files(label_dir, {".txt"})
        matched = bucket_missing = 0
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS or image_path.name.startswith("._"):
                continue
            mask_path = next(
                (mask_index[key] for key in candidate_stems(image_path.stem, "mask") if key in mask_index), None
            )
            label_path = next(
                (label_index[key] for key in candidate_stems(image_path.stem, "box") if key in label_index), None
            )
            if mask_path is None or label_path is None:
                bucket_missing += 1
                continue
            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label_path": label_path,
                    "dataset": dataset_name,
                    "size": size_bucket,
                    "stratum": f"{dataset_name}_{size_bucket}",
                }
            )
            matched += 1
        missing_files += bucket_missing
        print(f"{dataset_name}/{size_bucket}: {matched} matched, {bucket_missing} missing")
print("Total matched:", len(samples), "| missing:", missing_files)
if len(samples) != 12670:
    raise RuntimeError(f"Expected 12,670 image-mask-label triplets, found {len(samples)}.")

by_stratum = defaultdict(list)
for sample in samples:
    by_stratum[sample["stratum"]].append(sample)
split_rng = random.Random(SEED)
train_samples, val_samples, test_samples = [], [], []
for _, group in sorted(by_stratum.items()):
    group = list(group)
    split_rng.shuffle(group)
    train_end = int(len(group) * 0.70)
    val_end = train_end + int(len(group) * 0.15)
    train_samples.extend(group[:train_end])
    val_samples.extend(group[train_end:val_end])
    test_samples.extend(group[val_end:])
split_rng.shuffle(train_samples)
split_rng.shuffle(val_samples)
split_rng.shuffle(test_samples)
assert (len(train_samples), len(val_samples), len(test_samples)) == (8858, 1892, 1920)

if SMOKE_TEST:
    train_samples = train_samples[:8]
    val_samples = val_samples[:4]
    test_samples = test_samples[:4]


def describe_split(name, split):
    sizes = Counter(sample["size"] for sample in split)
    print(f"{name}: total={len(split)} small={sizes['small']} medium={sizes['medium']} large={sizes['large']}")


describe_split("Train", train_samples)
describe_split("Validation", val_samples)
describe_split("Test", test_samples)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_tensor(rgb_array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(rgb_array.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class DefectSegmentationDataset(Dataset):
    def __init__(self, split_samples, include_zoom=False):
        self.samples = split_samples
        self.include_zoom = include_zoom

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        mask_image = Image.open(sample["mask_path"]).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST
        )
        rgb = np.asarray(image).copy()
        mask = (np.asarray(mask_image) > 0).astype(np.uint8)
        components = connected_component_map(mask)
        output = {
            "pixel_values": image_tensor(rgb),
            "labels": torch.from_numpy(mask.copy()).long(),
            "boundary": torch.from_numpy(generate_boundary(mask)),
            "components": torch.from_numpy(components),
            "size": sample["size"],
            "dataset": sample["dataset"],
            "image_path": str(sample["image_path"]),
        }
        if not self.include_zoom:
            return output

        crop = choose_area_normalized_crop(
            mask,
            target_area_ratio=TARGET_CROP_AREA_RATIO,
            eligible_max_area_ratio=SMALL_COMPONENT_MAX_AREA_RATIO,
            context_scale=CROP_CONTEXT_SCALE,
            minimum_side=CROP_MINIMUM_SIDE,
            jitter_fraction=CROP_JITTER_FRACTION,
        )
        if crop is None:
            zoom_rgb = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            zoom_mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
            zoom_box = (0, 0, IMG_SIZE, IMG_SIZE)
            zoom_valid = False
            original_ratio = crop_ratio = 0.0
        else:
            zoom_rgb, zoom_mask = extract_zoom_arrays(rgb, mask, crop, CROP_SIZE)
            zoom_box = crop.box_xyxy
            zoom_valid = True
            original_ratio, crop_ratio = crop.original_area_ratio, crop.crop_area_ratio
        output.update(
            {
                "zoom_pixel_values": image_tensor(zoom_rgb),
                "zoom_labels": torch.from_numpy(zoom_mask.copy()).long(),
                "zoom_boundary": torch.from_numpy(generate_boundary(zoom_mask)),
                "zoom_box": torch.tensor(zoom_box, dtype=torch.long),
                "zoom_valid": zoom_valid,
                "zoom_original_area_ratio": original_ratio,
                "zoom_crop_area_ratio": crop_ratio,
            }
        )
        return output


def make_loader(split, *, shuffle=False, include_zoom=False, batch_size=BATCH_SIZE):
    return DataLoader(
        DefectSegmentationDataset(split, include_zoom=include_zoom),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        # The default 8,858/8 split ends in a valid batch of two. Drop only a
        # true singleton, which would make PIDNet's pooled BatchNorm undefined.
        drop_last=shuffle and len(split) % batch_size == 1,
    )


train_loader = make_loader(train_samples, shuffle=True, include_zoom=USE_ZOOM)
val_loader = make_loader(val_samples)
test_loader = make_loader(test_samples)

model = build_pidnet_s(num_classes=2, augment=True).to(DEVICE)
trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
total_parameters = sum(parameter.numel() for parameter in model.parameters())
print(f"PIDNet-S parameters: total={total_parameters:,}, trainable={trainable_parameters:,}")
wandb.config.update({"parameters": total_parameters, "pretrained_backbone_loaded": False})

semantic_loss = OhemCrossEntropy(ignore_label=255, threshold=0.9, min_kept=131072, weights=(0.4, 1.0))
boundary_loss = BoundaryLoss(coefficient=20.0)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

RUN_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def distillation_schedule(epoch: int) -> float:
    if epoch < DISTILLATION_START_EPOCH:
        return 0.0
    return min(1.0, (epoch - DISTILLATION_START_EPOCH + 1) / max(DISTILLATION_RAMP_EPOCHS, 1))


@torch.inference_mode()
def evaluate(loader, *, grouped=False, measure_inference=False):
    model.eval()
    accumulators = {"overall": SegmentationMetrics(MIN_PRED_COMPONENT_PIXELS)}
    inference_seconds = 0.0
    measured_images = 0
    if DEVICE.type == "cuda" and measure_inference:
        torch.cuda.reset_peak_memory_stats()

    for batch in loader:
        images = batch["pixel_values"].to(DEVICE, non_blocking=True)
        labels = batch["labels"]
        if measure_inference and DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(images)
        final_logits = F.interpolate(outputs[1], labels.shape[-2:], mode="bilinear", align_corners=False)
        if measure_inference and DEVICE.type == "cuda":
            torch.cuda.synchronize()
        if measure_inference:
            inference_seconds += time.perf_counter() - start
            measured_images += images.shape[0]

        predictions = final_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        targets = labels.numpy().astype(np.uint8)
        for index in range(len(predictions)):
            keys = ["overall"]
            if grouped:
                keys.extend([f"size:{batch['size'][index]}", f"dataset:{batch['dataset'][index]}"])
            for key in keys:
                accumulators.setdefault(key, SegmentationMetrics(MIN_PRED_COMPONENT_PIXELS))
                accumulators[key].update(predictions[index], targets[index])

    rows = []
    for key, accumulator in accumulators.items():
        metrics = accumulator.finalize()
        group_type, group_name = ("overall", "overall") if key == "overall" else key.split(":", 1)
        metrics.update({"group_type": group_type, "split": group_name})
        rows.append(metrics)
    overall = next(row for row in rows if row["group_type"] == "overall")
    overall["inference_time_ms_per_image"] = 1000 * inference_seconds / max(measured_images, 1)
    overall["throughput_images_per_second"] = measured_images / max(inference_seconds, 1e-12)
    overall["peak_inference_memory_mb"] = (
        torch.cuda.max_memory_allocated() / (1024**2) if DEVICE.type == "cuda" and measure_inference else float("nan")
    )
    return rows


def checkpoint_payload(epoch, best_epoch, best_dice, epochs_without_improvement):
    payload = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_dice": best_dice,
        "epochs_without_improvement": epochs_without_improvement,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "method": METHOD,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if DEVICE.type == "cuda":
        payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
    return payload


history_path = RUN_DIR / "training_history.csv"
last_checkpoint_path = RUN_DIR / "last_model.pt"
best_checkpoint_path = RUN_DIR / "best_model.pt"
history = []
best_epoch, best_dice, epochs_without_improvement, start_epoch = -1, -1.0, 0, 1
if last_checkpoint_path.exists() and history_path.exists():
    checkpoint = torch.load(last_checkpoint_path, map_location=DEVICE, weights_only=False)
    if checkpoint.get("method") != METHOD:
        raise RuntimeError("Refusing to resume a checkpoint created with a different METHOD.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    best_epoch = int(checkpoint["best_epoch"])
    best_dice = float(checkpoint["best_dice"])
    epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
    start_epoch = int(checkpoint["epoch"]) + 1
    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if DEVICE.type == "cuda" and "cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    history = pd.read_csv(history_path).to_dict("records")
    print(f"Resuming at epoch {start_epoch}; best epoch={best_epoch}, Dice={best_dice:.4f}")


for epoch in range(start_epoch, MAX_EPOCHS + 1):
    model.train()
    epoch_start = time.perf_counter()
    running = defaultdict(float)
    component_terms = zoom_samples = 0
    schedule = distillation_schedule(epoch)

    for batch_index, batch in enumerate(train_loader, start=1):
        images = batch["pixel_values"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        boundaries = batch["boundary"].to(DEVICE, non_blocking=True)
        components = batch["components"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            full_outputs = resize_pidnet_outputs(model(images), labels.shape[-2:])
            full_loss, full_parts = pidnet_loss(
                full_outputs, labels, boundaries, semantic_loss, boundary_loss
            )
            total_loss = full_loss
            component_loss = full_loss.sum() * 0.0
            crop_loss = full_loss.sum() * 0.0
            distillation_loss = full_loss.sum() * 0.0
            teacher_stats = {"valid_fraction": 0.0, "foreground_valid_fraction": 0.0}

            if USE_COMPONENT_LOSS:
                component_loss, component_count = component_balanced_recall_loss(
                    full_outputs[1],
                    components,
                    maximum_area_ratio=SMALL_COMPONENT_MAX_AREA_RATIO,
                )
                total_loss = total_loss + COMPONENT_LOSS_WEIGHT * component_loss
                component_terms += component_count

            if USE_ZOOM:
                valid_indices = torch.nonzero(batch["zoom_valid"], as_tuple=False).flatten()
                valid_indices = valid_indices[:MAX_ZOOM_CROPS_PER_BATCH]
                if valid_indices.numel() > 0:
                    unique_zoom_samples = int(valid_indices.numel())
                    # PIDNet's global pooled PAPPM path contains BatchNorm on a 1x1 map;
                    # duplicate a singleton crop so training never presents B*H*W == 1.
                    if valid_indices.numel() == 1:
                        valid_indices = torch.cat([valid_indices, valid_indices])
                    valid_device = valid_indices.to(DEVICE)
                    zoom_images = batch["zoom_pixel_values"][valid_indices].to(DEVICE, non_blocking=True)
                    zoom_labels = batch["zoom_labels"][valid_indices].to(DEVICE, non_blocking=True)
                    zoom_boundaries = batch["zoom_boundary"][valid_indices].to(DEVICE, non_blocking=True)
                    zoom_boxes = batch["zoom_box"][valid_indices].to(DEVICE, non_blocking=True)
                    zoom_outputs = resize_pidnet_outputs(model(zoom_images), zoom_labels.shape[-2:])
                    crop_loss, _ = pidnet_loss(
                        zoom_outputs, zoom_labels, zoom_boundaries, semantic_loss, boundary_loss
                    )
                    total_loss = total_loss + CROP_SUPERVISION_WEIGHT * crop_loss
                    if schedule > 0:
                        distillation_loss, teacher_stats = quality_gated_zoom_distillation_loss(
                            full_outputs[1][valid_device],
                            zoom_outputs[1],
                            zoom_boxes,
                            zoom_labels,
                            temperature=DISTILLATION_TEMPERATURE,
                            minimum_teacher_confidence=TEACHER_CONFIDENCE,
                            foreground_weight=FOREGROUND_DISTILLATION_WEIGHT,
                        )
                        total_loss = total_loss + schedule * DISTILLATION_WEIGHT * distillation_loss
                    zoom_samples += unique_zoom_samples

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}, batch {batch_index}.")
        scaler.step(optimizer)
        scaler.update()

        running["total_loss"] += float(total_loss.detach())
        running["full_loss"] += float(full_loss.detach())
        running["component_loss"] += float(component_loss.detach())
        running["crop_loss"] += float(crop_loss.detach())
        running["distillation_loss"] += float(distillation_loss.detach())
        running["teacher_valid_fraction"] += teacher_stats["valid_fraction"]
        running["teacher_foreground_valid_fraction"] += teacher_stats["foreground_valid_fraction"]
        running["zoom_original_area_ratio"] += float(batch.get("zoom_original_area_ratio", torch.zeros(1)).sum())
        running["zoom_crop_area_ratio"] += float(batch.get("zoom_crop_area_ratio", torch.zeros(1)).sum())

        if batch_index % PROGRESS_EVERY == 0 or batch_index == len(train_loader):
            elapsed = time.perf_counter() - epoch_start
            print(
                f"  epoch {epoch:02d} batch {batch_index}/{len(train_loader)} "
                f"loss={running['total_loss']/batch_index:.4f} zoom={zoom_samples} "
                f"elapsed={elapsed/60:.1f}m eta={(elapsed/batch_index)*(len(train_loader)-batch_index)/60:.1f}m",
                flush=True,
            )

    validation_rows = evaluate(val_loader)
    validation = next(row for row in validation_rows if row["group_type"] == "overall")
    batches = max(len(train_loader), 1)
    row = {
        "epoch": epoch,
        "train_total_loss": running["total_loss"] / batches,
        "train_full_loss": running["full_loss"] / batches,
        "train_component_loss": running["component_loss"] / batches,
        "train_crop_loss": running["crop_loss"] / batches,
        "train_distillation_loss": running["distillation_loss"] / batches,
        "distillation_schedule": schedule,
        "teacher_valid_fraction": running["teacher_valid_fraction"] / batches,
        "teacher_foreground_valid_fraction": running["teacher_foreground_valid_fraction"] / batches,
        "zoom_samples": zoom_samples,
        "component_loss_terms": component_terms,
        "val_precision": validation["precision"],
        "val_recall": validation["recall"],
        "val_iou": validation["iou"],
        "val_dice": validation["dice"],
        "val_fp_per_image": validation["fp_per_image"],
        "val_component_recall_iou10": validation["component_recall_iou10"],
        "epoch_minutes": (time.perf_counter() - epoch_start) / 60,
    }
    history.append(row)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={row['train_total_loss']:.4f} "
        f"| val Dice={row['val_dice']:.4f} IoU={row['val_iou']:.4f} "
        f"Recall={row['val_recall']:.4f} component-R@0.1={row['val_component_recall_iou10']:.4f}",
        flush=True,
    )
    wandb.log(row, step=epoch)

    improved = row["val_dice"] > best_dice
    if improved:
        best_dice = row["val_dice"]
        best_epoch = epoch
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    payload = checkpoint_payload(epoch, best_epoch, best_dice, epochs_without_improvement)
    torch.save(payload, last_checkpoint_path)
    if improved:
        torch.save(payload, best_checkpoint_path)
    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}; best Dice={best_dice:.4f} at epoch {best_epoch}.")
        break

if not best_checkpoint_path.exists():
    raise RuntimeError("Training ended without producing a best checkpoint.")
checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
wandb.summary["best_epoch"] = best_epoch
wandb.summary["best_validation_dice"] = best_dice

test_rows = evaluate(test_loader, grouped=True, measure_inference=True)
test_df = pd.DataFrame(test_rows)
test_df.to_csv(FINAL_OUTPUT_DIR / "evaluation_metrics.csv", index=False)
print(test_df.to_string(index=False))
for test_row in test_rows:
    prefix = f"test/{test_row['group_type']}/{test_row['split']}"
    wandb.log(
        {
            f"{prefix}/{key}": value
            for key, value in test_row.items()
            if isinstance(value, (int, float))
            and not (isinstance(value, float) and math.isnan(value))
        }
    )


@torch.inference_mode()
def save_prediction_examples(samples_per_size=3):
    """Save deterministic original/GT/prediction/error panels for qualitative QA."""
    selected = []
    for size in SIZE_BUCKETS:
        selected.extend([sample for sample in test_samples if sample["size"] == size][:samples_per_size])
    output_dir = FINAL_OUTPUT_DIR / "prediction_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    model.eval()
    for sample in selected:
        image = Image.open(sample["image_path"]).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        target = Image.open(sample["mask_path"]).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST
        )
        rgb = np.asarray(image).copy()
        target_array = np.asarray(target) > 0
        inputs = image_tensor(rgb).unsqueeze(0).to(DEVICE)
        outputs = model(inputs)
        logits = F.interpolate(outputs[1], (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0].cpu().numpy() == 1

        ground_truth_panel = rgb.copy()
        ground_truth_panel[target_array] = (
            0.35 * ground_truth_panel[target_array] + 0.65 * np.array([0, 255, 0])
        ).astype(np.uint8)
        prediction_panel = rgb.copy()
        prediction_panel[prediction] = (
            0.35 * prediction_panel[prediction] + 0.65 * np.array([255, 180, 0])
        ).astype(np.uint8)
        error_panel = (rgb * 0.30).astype(np.uint8)
        true_positive = prediction & target_array
        false_positive = prediction & ~target_array
        false_negative = ~prediction & target_array
        error_panel[true_positive] = [0, 220, 0]
        error_panel[false_positive] = [255, 0, 0]
        error_panel[false_negative] = [0, 120, 255]

        canvas = Image.fromarray(np.concatenate([rgb, ground_truth_panel, prediction_panel, error_panel], axis=1))
        draw = ImageDraw.Draw(canvas)
        for panel_index, label in enumerate(
            ["Original", "Ground truth", "Prediction", "Error: TP green / FP red / FN blue"]
        ):
            x = panel_index * IMG_SIZE
            draw.rectangle((x, 0, x + min(280, IMG_SIZE), 20), fill=(0, 0, 0))
            draw.text((x + 4, 3), label, fill=(255, 255, 255))
        safe_dataset = sample["dataset"].replace("/", "-")
        destination = output_dir / f"{sample['size']}_{safe_dataset}_{sample['image_path'].stem}.png"
        canvas.save(destination)
        saved.append(destination)
    if saved:
        wandb.log({"prediction_examples": [wandb.Image(str(path)) for path in saved]})
    return saved


prediction_examples = save_prediction_examples(samples_per_size=1 if SMOKE_TEST else 3)


@torch.inference_mode()
def benchmark_batch_one(repetitions=100, warmups=20):
    loader = make_loader(test_samples[:1], batch_size=1)
    sample = next(iter(loader))["pixel_values"].to(DEVICE)
    model.eval()
    for _ in range(warmups):
        output = model(sample)
        F.interpolate(output[1], (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        output = model(sample)
        F.interpolate(output[1], (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repetitions


batch_one_latency_ms = benchmark_batch_one(repetitions=10 if SMOKE_TEST else 100, warmups=2 if SMOKE_TEST else 20)
@torch.inference_mode()
def count_convolution_macs():
    """Count Conv2d MACs with hooks, avoiding profiler-version dependencies."""
    total = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal total
        batch, output_channels, output_height, output_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        per_output = (module.in_channels // module.groups) * kernel_height * kernel_width
        total += batch * output_channels * output_height * output_width * per_output

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    was_training = model.training
    model.eval()
    model(dummy)
    model.train(was_training)
    for registered_hook in hooks:
        registered_hook.remove()
    return total


macs = count_convolution_macs()

rows_by_key = {(row["group_type"], row["split"]): row for row in test_rows}
overall = rows_by_key[("overall", "overall")]
summary = {
    "Experiment": RUN_NAME,
    "Model": MODEL_LABEL,
    "Method": METHOD,
    "Batch": BATCH_SIZE,
    "Epochs": best_epoch,
    "mAP50": float("nan"),
    "mAP50_95": float("nan"),
    "Precision": overall["precision"],
    "Recall": overall["recall"],
    "mAP50_Small": float("nan"),
    "mAP50_Medium": float("nan"),
    "mAP50_Large": float("nan"),
    "Recall_Small": rows_by_key[("size", "small")]["recall"],
    "Recall_Medium": rows_by_key[("size", "medium")]["recall"],
    "Recall_Large": rows_by_key[("size", "large")]["recall"],
    "Inference_Time_ms": overall["inference_time_ms_per_image"],
    "Latency_Batch1_ms": batch_one_latency_ms,
    "Throughput_Images_per_s": overall["throughput_images_per_second"],
    "Peak_Inference_Memory_MB": overall["peak_inference_memory_mb"],
    "FP_per_Image": overall["fp_per_image"],
    "Dice": overall["dice"],
    "IoU": overall["iou"],
    "Specificity": overall["specificity"],
    "Accuracy": overall["accuracy"],
    "Component_Recall_IoU10": overall["component_recall_iou10"],
    "Component_Precision_IoU10": overall["component_precision_iou10"],
    "Component_F1_IoU10": overall["component_f1_iou10"],
    "Component_Recall_IoU50": overall["component_recall_iou50"],
    "Component_Precision_IoU50": overall["component_precision_iou50"],
    "Component_F1_IoU50": overall["component_f1_iou50"],
    "Mean_Best_Component_IoU": overall["mean_best_component_iou"],
    "Parameters": total_parameters,
    "Trainable_Parameters": trainable_parameters,
    "GMACs_640": macs / 1e9 if IMG_SIZE == 640 else float("nan"),
    "GMACs_Profiled": macs / 1e9,
    "Profile_Input_Size": IMG_SIZE,
    "Notes": (
        "Segmentation-only PIDNet-S; area-normalized zoom branch and component loss are training-only; "
        f"deployed architecture is unchanged. Fixed 70/15/15 size-stratified split, seed {SEED}, "
        f"{IMG_SIZE} input. "
        "mAP intentionally blank because this is binary semantic segmentation."
    ),
}
pd.DataFrame([summary]).to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)

metadata = {
    "experiment": RUN_NAME,
    "model": MODEL_LABEL,
    "method": METHOD,
    "task": "binary semantic defect segmentation",
    "deployed_architecture_changed": False,
    "pretrained_backbone_loaded": False,
    "configuration": dict(wandb.config),
    "best_epoch": best_epoch,
    "best_validation_dice": best_dice,
    "split_counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
    "metric_definitions": {
        "FP_per_Image": "false-positive pixels divided by images, matching the existing segmentation baselines",
        "component_metrics": (
            f"8-connected predicted components of at least {MIN_PRED_COMPONENT_PIXELS} pixels, greedily matched "
            "one-to-one to ground-truth components at the stated IoU threshold"
        ),
        "Inference_Time_ms": "model forward plus final bilinear upsampling; data loading and metric computation excluded",
    },
    "prediction_examples": [str(path) for path in prediction_examples],
}
(FINAL_OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

for source, destination in (
    (best_checkpoint_path, FINAL_OUTPUT_DIR / "best_model.pt"),
    (last_checkpoint_path, FINAL_OUTPUT_DIR / "last_model.pt"),
    (history_path, FINAL_OUTPUT_DIR / "training_history.csv"),
):
    shutil.copy2(source, destination)

wandb.log(
    {
        f"test/{key}": value
        for key, value in summary.items()
        if isinstance(value, (int, float))
        and not (isinstance(value, float) and math.isnan(value))
    }
)
for key, value in summary.items():
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        wandb.summary[key] = value
wandb.finish()
print("Saved final outputs to:", FINAL_OUTPUT_DIR)
print(pd.DataFrame([summary]).to_string(index=False))
print("DONE")
