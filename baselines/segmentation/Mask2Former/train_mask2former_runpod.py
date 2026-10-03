"""
Mask2Former (Swin-Tiny, ADE20K-semantic pretrained), binary defect segmentation -- RunPod version.

Faithful port of baselines/segmentation/Mask2Former/mask2former_kaggle.ipynb: same
HuggingFace checkpoint, same dataset split, same differential learning rate (verified
against this checkpoint's real parameter names), same full-precision training (AMP
deliberately off -- DETR-style Hungarian-matching loss, same architecture risk class
as D-FINE's documented AMP NaN failure on this dataset), same D-FINE-matching
summary.csv schema, same per-batch progress logging. Moved here because Kaggle's free
T4 pegs at 100% GPU / ~109% CPU (confirmed via the resource monitor, not a bug -- this
model is just genuinely heavy: no AMP, batch 4, Swin backbone + pixel/transformer
decoder, and HuggingFace's Hungarian matching runs on CPU via scipy, not GPU) without
finishing even one epoch in under an hour. Only the paths changed (Kaggle -> RunPod)
and the dataset now comes from the same local-tar-or-HF-download pattern as every
other RunPod script in this project.

Run:
    export WANDB_API_KEY=<your wandb key>       # optional, else interactive login prompt
    export HF_TOKEN=<your huggingface token>    # only needed if DATASET_ROOT isn't already populated
    export DATASET_ROOT=/workspace/dataset      # optional, this is the default
    nohup python -u train_mask2former_runpod.py > train.log 2>&1 &
    tail -f train.log

Same params as the Kaggle notebook, unchanged: batch 4, 50 max epochs / patience 15,
differential LR (backbone 1e-5 / head 1e-4), weight decay 0.05, full precision (no AMP).
If you land on a large-VRAM GPU (24GB+), batch size can likely go higher than 4 -- that
wasn't changed here to keep this a controlled, faithful port; bump BATCH_SIZE yourself
if you want to use the extra headroom.
"""

import os

# Defensive, same reasoning as every other RunPod script in this project.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# Config -- same values as mask2former_kaggle.ipynb
# ============================================================
RUN_NAME = "RunM_mask2former_swin_tiny_imgsz640"
MODEL_NAME = "facebook/mask2former-swin-tiny-ade-semantic"
MODEL_LABEL = "Mask2Former"
WANDB_PROJECT = "smallDefectDetection"
WANDB_RUN_NAME = f"{MODEL_LABEL}_segmentation"

HF_DATASET_REPO = "Smalldefect/SmallDefectDataseet"  # private -- needs HF_TOKEN if downloading fresh

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMG_SIZE = 640
BATCH_SIZE = 4
MAX_EPOCHS = 50
PATIENCE = 15
BASE_LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 0.05
NUM_WORKERS = 2

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/mask2former_run"))
RUN_DIR = BASE_DIR / "mask2former_runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

random.seed(SEED)

# ============================================================
# numpy/pandas/Pillow/wandb/transformers are not guaranteed preinstalled on every
# RunPod base image (confirmed the hard way on the D-FINE run -- pandas was missing).
# ============================================================
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
     "numpy", "pandas", "pillow", "wandb", "transformers>=4.51.0,<4.52.0", "safetensors", "scipy"],
    check=True,
)

# ============================================================
# CUDA / numpy sanity check.
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

print("torch:", torch.__version__, "| torch's CUDA build:", torch.version.cuda)
print("numpy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count(), "(should be 1)")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

_smoke = torch.from_numpy(np.zeros(3, dtype=np.float32))
if torch.cuda.is_available():
    _smoke = (_smoke.cuda() + 1).cpu()
print("numpy <-> torch <-> CUDA smoke test passed.")

if not torch.cuda.is_available():
    sys.exit("CUDA is not available on this pod. Check the GPU is attached and the driver/CUDA toolkit is loaded (`nvidia-smi`).")
if torch.cuda.device_count() != 1:
    sys.exit(f"Expected exactly 1 visible GPU, found {torch.cuda.device_count()} despite CUDA_VISIBLE_DEVICES=0 -- investigate before continuing.")

DEVICE = torch.device("cuda")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def wandb_login_anywhere():
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            api_key = None
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()


wandb_login_anywhere()
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "model": MODEL_NAME,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "base_lr": BASE_LR,
        "backbone_lr": BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "amp": False,
    },
)

# ============================================================
# Dataset: same download-if-missing pattern as train_dfine_s_runpod.py /
# train_segnext_t_runpod.py -- checks for images/masks/labels_yolo dirs, downloads
# from the private HF repo with retry-with-backoff if not already present. Skips
# entirely if you scp'd + untarred dataset.tar to DATASET_ROOT beforehand.
# ============================================================
def missing_dataset_dirs():
    return [
        f"{name}/{size}"
        for name in DATASET_NAMES
        for size in SIZE_BUCKETS
        if not (DATASET_ROOT / name / size / "images").exists()
        or not (DATASET_ROOT / name / size / "masks").exists()
        or not (DATASET_ROOT / name / size / "labels_yolo").exists()
    ]


print(f"Checking dataset at {DATASET_ROOT} ...")
missing = missing_dataset_dirs()

if missing:
    print(f"Dataset incomplete/missing at {DATASET_ROOT} ({len(missing)} "
          f"bucket(s) missing) -- attempting download from {HF_DATASET_REPO}.")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "HF_TOKEN is not set and the dataset isn't already on this pod. "
            "The dataset repo is private, so downloading it requires a token:\n"
            "  export HF_TOKEN=<your huggingface token>\n"
            "then re-run. Alternatively, scp a pre-built dataset.tar to this pod "
            "and extract it to DATASET_ROOT yourself before running this script -- "
            "it'll skip the download entirely if the expected structure is already there."
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "--no-cache-dir", "huggingface_hub"],
        check=True,
    )
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
    from huggingface_hub import snapshot_download

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_RETRIES = 6
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir=str(DATASET_ROOT),
                token=hf_token,
                max_workers=4,
            )
            break
        except Exception as error:
            if attempt == DOWNLOAD_RETRIES:
                raise
            wait_seconds = 30 * attempt
            print(f"Download attempt {attempt}/{DOWNLOAD_RETRIES} failed ({error}); "
                  f"waiting {wait_seconds}s and retrying. snapshot_download resumes "
                  "from already-fetched files, it doesn't restart from zero.")
            time.sleep(wait_seconds)
    print("Download finished. Re-checking structure...")
    missing = missing_dataset_dirs()

if missing:
    sys.exit(
        f"Dataset still incomplete at {DATASET_ROOT}. Missing images/masks/labels_yolo under: "
        f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
    )
print("Dataset structure looks complete.")

# ============================================================
# Match images + masks + labels (identical logic to mask2former_kaggle.ipynb)
# ============================================================
def index_files(directory, suffixes):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def candidate_stems(image_stem, target):
    base = image_stem.removesuffix("_defect")
    if target == "mask":
        return [image_stem, image_stem.replace("_defect", "_mask"), base, base + "_mask", base + "_gt"]
    return [image_stem, image_stem.replace("_defect", "_bbs"), base, base + "_bbs"]


samples = []
missing_count = 0
for dataset_name in DATASET_NAMES:
    for size_bucket in SIZE_BUCKETS:
        bucket_root = DATASET_ROOT / dataset_name / size_bucket
        image_dir = bucket_root / "images"
        mask_dir = bucket_root / "masks"
        label_dir = bucket_root / "labels_yolo"

        mask_index = index_files(mask_dir, IMAGE_EXTS)
        label_index = index_files(label_dir, {".txt"})
        matched = 0
        bucket_missing = 0

        for image_path in image_dir.iterdir():
            # Skip macOS AppleDouble metadata sidecars (._<name>.<ext>) -- same
            # bug this project already hit once, doubling the dataset with junk.
            if image_path.suffix.lower() not in IMAGE_EXTS or image_path.name.startswith("._"):
                continue
            mask_path = next((mask_index[s] for s in candidate_stems(image_path.stem, "mask") if s in mask_index), None)
            label_path = next((label_index[s] for s in candidate_stems(image_path.stem, "box") if s in label_index), None)
            if mask_path is None or label_path is None:
                bucket_missing += 1
                continue
            samples.append({
                "image_path": image_path, "mask_path": mask_path,
                "dataset": dataset_name, "size": size_bucket,
                "stratum": f"{dataset_name}_{size_bucket}",
            })
            matched += 1
        missing_count += bucket_missing
        print(f"{dataset_name}/{size_bucket}: {matched} matched, {bucket_missing} missing")

print("Total matched:", len(samples), "| missing:", missing_count)
if len(samples) != 12670:
    sys.exit(f"Expected 12,670 image-mask-label triplets, found {len(samples)}.")

# ============================================================
# Stratified split (identical seed/ratios to every other script here)
# ============================================================
by_stratum = defaultdict(list)
for sample in samples:
    by_stratum[sample["stratum"]].append(sample)

rng = random.Random(SEED)
train_samples, val_samples, test_samples = [], [], []
for _, group in sorted(by_stratum.items()):
    group = list(group)
    rng.shuffle(group)
    train_end = int(len(group) * TRAIN_RATIO)
    val_end = train_end + int(len(group) * VAL_RATIO)
    train_samples.extend(group[:train_end])
    val_samples.extend(group[train_end:val_end])
    test_samples.extend(group[val_end:])

rng.shuffle(train_samples)
rng.shuffle(val_samples)
rng.shuffle(test_samples)

test_sets = {
    "overall": test_samples,
    "small": [s for s in test_samples if s["size"] == "small"],
    "medium": [s for s in test_samples if s["size"] == "medium"],
    "large": [s for s in test_samples if s["size"] == "large"],
}


def print_counts(name, split):
    counts = Counter(s["size"] for s in split)
    print(f"{name}: total={len(split)} small={counts['small']} medium={counts['medium']} large={counts['large']}")


print_counts("Train", train_samples)
print_counts("Validation", val_samples)
for name, split in test_sets.items():
    print_counts("Test " + name, split)

assert len(train_samples) == 8858
assert len(val_samples) == 1892
assert len(test_samples) == 1920

def load_binary_mask(mask_path, target_size):
    mask = Image.open(mask_path).convert('L')
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


# do_resize=False: images/masks are resized ourselves below, so the exact same array
# that produces mask_labels/class_labels (via the processor) is also kept as the raw
# ground-truth tensor for pixel-level metrics -- no risk of the processor's internal
# resize logic silently drifting from what's used for evaluation.
# do_reduce_labels=False: label 0 is a real 'background' class here, not 'ignore' --
# ADE20K's own convention (which this checkpoint was pretrained under) would otherwise
# treat class 0 as ignore and shift everything down by one, which is wrong for us.
processor = Mask2FormerImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=False,
    do_reduce_labels=False,
)


class DefectMaskDataset(Dataset):
    def __init__(self, split_samples):
        self.samples = split_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample['image_path']).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        mask = load_binary_mask(sample['mask_path'], (IMG_SIZE, IMG_SIZE))
        encoded = processor(images=image, segmentation_maps=mask, return_tensors='pt')
        return {
            'pixel_values': encoded['pixel_values'][0],
            'pixel_mask': encoded['pixel_mask'][0],
            'mask_labels': encoded['mask_labels'][0],
            'class_labels': encoded['class_labels'][0],
            'gt_semantic_mask': torch.from_numpy(mask).long(),
        }


def collate_fn(batch):
    # mask_labels/class_labels are variable-length per image (however many classes are
    # actually present) -- they can't be torch.stack'd like a normal batch, Mask2Former's
    # forward() expects them as plain lists. Verified this exact shape against the real
    # model API locally before using it here.
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'pixel_mask': torch.stack([item['pixel_mask'] for item in batch]),
        'mask_labels': [item['mask_labels'] for item in batch],
        'class_labels': [item['class_labels'] for item in batch],
        'gt_semantic_mask': torch.stack([item['gt_semantic_mask'] for item in batch]),
    }


def make_loader(split_samples, shuffle=False):
    return DataLoader(
        DefectMaskDataset(split_samples),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=collate_fn,
    )


train_loader = make_loader(train_samples, shuffle=True)
val_loader = make_loader(val_samples)
print('Train batches:', len(train_loader), 'Validation batches:', len(val_loader))

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
).to(DEVICE)

# Differential LR: lower for the pretrained Swin backbone, higher for the randomly
# reinitialized classification/mask heads -- Mask2Former's own training recipe uses this
# split, and 'pixel_level_module.encoder' was confirmed against this checkpoint's real
# parameter names (431 backbone / 328 head params) before writing this, not guessed.
backbone_params = [p for n, p in model.named_parameters() if 'pixel_level_module.encoder' in n]
head_params = [p for n, p in model.named_parameters() if 'pixel_level_module.encoder' not in n]
optimizer = AdamW([
    {'params': backbone_params, 'lr': BACKBONE_LR},
    {'params': head_params, 'lr': BASE_LR},
], weight_decay=WEIGHT_DECAY)

RUN_DIR.mkdir(parents=True, exist_ok=True)
print('Model loaded. Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(f'Backbone params: {sum(p.numel() for p in backbone_params):,} (lr={BACKBONE_LR}), head params: {sum(p.numel() for p in head_params):,} (lr={BASE_LR})')

@torch.inference_mode()
def evaluate(loader, measure_inference=False):
    model.eval()
    true_positive = false_positive = false_negative = 0
    inference_seconds = 0.0
    image_count = 0

    for batch in loader:
        pixel_values = batch['pixel_values'].to(DEVICE, non_blocking=True)
        pixel_mask = batch['pixel_mask'].to(DEVICE, non_blocking=True)
        gt_masks = batch['gt_semantic_mask']

        if measure_inference:
            torch.cuda.synchronize()
            start = time.perf_counter()
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if measure_inference:
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start

        target_sizes = [(IMG_SIZE, IMG_SIZE)] * pixel_values.shape[0]
        predictions = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        predictions = torch.stack(predictions).cpu()

        true_positive += int(((predictions == 1) & (gt_masks == 1)).sum().item())
        false_positive += int(((predictions == 1) & (gt_masks == 0)).sum().item())
        false_negative += int(((predictions == 0) & (gt_masks == 1)).sum().item())
        image_count += gt_masks.shape[0]

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)

    return {
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'dice': dice,
        # Pixel-level FPs per image, not per-instance -- kept identical in spirit to
        # SegFormer/SegNeXt's fp_per_image even though this model is instance-capable,
        # since it's being evaluated in semantic mode here (see markdown cell).
        'fp_per_image': false_positive / max(image_count, 1),
        'inference_time_ms_per_image': 1000 * inference_seconds / max(image_count, 1),
        'images': image_count,
    }


history = []
best_dice = -1.0
best_epoch = -1
epochs_without_improvement = 0
start_epoch = 1
PROGRESS_EVERY = 50  # batches -- this loop used to print nothing until a full epoch
                      # finished, which on a slow GPU looks indistinguishable from a hang.

# Resume support: if this exact run was interrupted (pod stopped, crashed, OOM, etc.)
# and left a checkpoint + history behind on the persistent volume, pick up from there
# instead of silently restarting from the original pretrained weights and losing
# whatever wall-clock time was already spent. Added after a real pod-stop mid-training
# with no way to recover 29 epochs of progress on the first version of this script.
history_path = RUN_DIR / 'training_history.csv'
checkpoint_path = RUN_DIR / 'best_model.pt'
if history_path.exists() and checkpoint_path.exists():
    print(f'Found existing checkpoint + history at {RUN_DIR} -- resuming instead of restarting.')
    resume_checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(resume_checkpoint['model_state_dict'])
    optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
    best_epoch = resume_checkpoint['epoch']
    best_dice = resume_checkpoint['val_metrics']['dice']

    history_df = pd.read_csv(history_path)
    history = history_df.to_dict('records')
    last_logged_epoch = int(history_df['epoch'].max())
    start_epoch = last_logged_epoch + 1
    epochs_without_improvement = last_logged_epoch - best_epoch
    print(f'Resuming from epoch {start_epoch} (last logged epoch: {last_logged_epoch}). '
          f'Best so far: epoch {best_epoch}, val Dice={best_dice:.4f}, '
          f'{epochs_without_improvement} epoch(s) without improvement.')
    if epochs_without_improvement >= PATIENCE:
        print(f'WARNING: already at/past patience ({epochs_without_improvement} >= {PATIENCE}) '
              'as of the last logged epoch -- this run will likely stop almost immediately. '
              'That is correct behavior if it truly plateaued, not a bug.')
else:
    print('No existing checkpoint found -- starting fresh from the original pretrained weights.')

for epoch in range(start_epoch, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    epoch_start = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader, start=1):
        pixel_values = batch['pixel_values'].to(DEVICE, non_blocking=True)
        pixel_mask = batch['pixel_mask'].to(DEVICE, non_blocking=True)
        mask_labels = [m.to(DEVICE, non_blocking=True) for m in batch['mask_labels']]
        class_labels = [c.to(DEVICE, non_blocking=True) for c in batch['class_labels']]

        optimizer.zero_grad(set_to_none=True)
        # No AMP here on purpose -- see markdown cell. Full precision only.
        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += float(loss.item())

        if batch_idx % PROGRESS_EVERY == 0 or batch_idx == len(train_loader):
            elapsed = time.perf_counter() - epoch_start
            rate = elapsed / batch_idx
            eta = rate * (len(train_loader) - batch_idx)
            print(f"  epoch {epoch:02d} batch {batch_idx}/{len(train_loader)} "
                  f"| avg loss={running_loss / batch_idx:.4f} "
                  f"| {rate:.2f}s/batch | elapsed={elapsed/60:.1f}m | ETA this epoch={eta/60:.1f}m",
                  flush=True)

    val_metrics = evaluate(val_loader)
    row = {'epoch': epoch, 'train_loss': running_loss / len(train_loader), **val_metrics}
    history.append(row)
    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={row['train_loss']:.4f} | val Dice={row['dice']:.4f} | val IoU={row['iou']:.4f} | val Recall={row['recall']:.4f}", flush=True)
    wandb.log(row, step=epoch)

    if row['dice'] > best_dice:
        best_dice = row['dice']
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
        }, RUN_DIR / 'best_model.pt')
    else:
        epochs_without_improvement += 1

    pd.DataFrame(history).to_csv(RUN_DIR / 'training_history.csv', index=False)
    if epochs_without_improvement >= PATIENCE:
        print(f'Early stopping at epoch {epoch}; best validation Dice was {best_dice:.4f} at epoch {best_epoch}.')
        break

wandb.summary['best_epoch'] = best_epoch
wandb.summary['best_validation_dice'] = best_dice
print('Best epoch:', best_epoch, 'Best validation Dice:', best_dice)

# ============================================================
# Load the best validation-Dice checkpoint and evaluate every fixed test subset.
# ============================================================
checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

test_rows = []
for split_name, split_samples in test_sets.items():
    metrics = evaluate(make_loader(split_samples), measure_inference=True)
    test_rows.append({"split": split_name, **metrics})
    print(split_name, metrics)
    wandb.log({f"test_{split_name}_{key}": value for key, value in metrics.items()})

test_df = pd.DataFrame(test_rows)
print(test_df.to_string(index=False))

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(RUN_DIR / "best_model.pt", FINAL_OUTPUT_DIR / "best_model.pt")
shutil.copy2(RUN_DIR / "training_history.csv", FINAL_OUTPUT_DIR / "training_history.csv")
# Long-format, one row per split -- same shape as D-FINE's evaluation_metrics.csv.
test_df.to_csv(FINAL_OUTPUT_DIR / "evaluation_metrics.csv", index=False)

by_split = {row["split"]: row for row in test_rows}
overall = by_split["overall"]

# Wide-format, one row per experiment -- matches the D-FINE/detection summary.csv schema
# exactly. mAP left blank: Mask2Former is instance-capable in general, but this run
# evaluates it in binary semantic mode for consistency with SegFormer/SegNeXt (see
# feedback_segmentation_metrics_schema in project memory for why).
summary_row = {
    "Experiment": RUN_NAME,
    "Model": MODEL_LABEL,
    "Batch": BATCH_SIZE,
    "Epochs": best_epoch,
    "mAP50": float("nan"),
    "mAP50_95": float("nan"),
    "Precision": overall["precision"],
    "Recall": overall["recall"],
    "mAP50_Small": float("nan"),
    "mAP50_Medium": float("nan"),
    "mAP50_Large": float("nan"),
    "Recall_Small": by_split["small"]["recall"],
    "Recall_Medium": by_split["medium"]["recall"],
    "Recall_Large": by_split["large"]["recall"],
    "Inference_Time_ms": overall["inference_time_ms_per_image"],
    "FP_per_Image": overall["fp_per_image"],
    "Dice": overall["dice"],
    "IoU": overall["iou"],
    "Notes": f"{MODEL_LABEL} via HuggingFace transformers ({MODEL_NAME}) on RunPod, evaluated in binary semantic segmentation mode (not instance mode) for consistency with SegFormer/SegNeXt, full precision (no AMP -- DETR-style Hungarian matching, same risk class as D-FINE's documented AMP NaN issue), differential LR, fixed 640 split. mAP intentionally blank -- not computed in this semantic-mode run.",
}
summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)

metadata = {
    "experiment": RUN_NAME,
    "model": MODEL_LABEL,
    "task": "binary semantic defect segmentation",
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "base_lr": BASE_LR,
    "backbone_lr": BACKBONE_LR,
    "amp": False,
    "best_epoch": best_epoch,
    "best_validation_dice": best_dice,
    "split_counts": {
        "train": len(train_samples), "val": len(val_samples),
        "test": len(test_samples), "test_small": len(test_sets["small"]),
        "test_medium": len(test_sets["medium"]), "test_large": len(test_sets["large"]),
    },
}
(FINAL_OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

print("Saved final artifacts to:", FINAL_OUTPUT_DIR)
print(summary_df.to_string(index=False))
wandb.finish()
print("DONE.")
