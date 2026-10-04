"""LiteJointModel training: shared Swin-Tiny backbone + a new YOLO-style anchor-free (TAL/DFL)
detection head + the existing, unmodified Mask2Former segmentation path. Replaces DefectFormer's
DETR-style detection decoder (root-caused across v1-v4 to Hungarian-matching instability) with a
paradigm that doesn't depend on bipartite matching at all. See novelty_experiments/lite_joint_model.py
and the approved plan (2026-08-20/21 session) for the full architecture rationale.

Two genuinely separate stages, not phases inside one run -- see STAGE below:
  STAGE=1 (default): detection-only smoke test. Backbone frozen, segmentation loss off, a
    handful of epochs. Gate: real mAP50 (not just a decreasing loss -- that's exactly what
    v4's denoising already showed without beating baseline) + non-degenerate boxes + measured
    peak memory. Do not proceed to Stage 2 without this passing.
  STAGE=2: full joint run, backbone unfrozen with differential LR, full budget, evaluated
    against every real baseline in baselines_small_defect_detection.pdf.

Run:
    export WANDB_API_KEY=<your wandb key>
    export DATASET_ROOT=/workspace/dataset
    export STAGE=1
    nohup python -u train_lightweight_runpod.py > train_stage1.log 2>&1 &
    tail -f train_stage1.log

novelty_experiments/*.py must sit next to this script on the pod (scp the whole
novelty_experiments/ folder alongside the dataset).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# novelty_experiments/ itself holds detection_metrics.py, shared with defectformer/
# (train_novelty_runpod.py's model); lite_joint/ holds this model's own code.
sys.path.insert(0, str(Path(__file__).parent.parent / "defectformer"))
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Config
# ============================================================
STAGE = int(os.environ.get("STAGE", "1"))
assert STAGE in (1, 2), f"STAGE must be 1 or 2, got {STAGE}"

RUN_NAME = f"RunT_lite_joint_stage{STAGE}_imgsz640"
MODEL_LABEL = f"LiteJoint (Swin-Tiny shared backbone + TAL/DFL detect head + Mask2Former seg) -- stage {STAGE}"
BACKBONE_CHECKPOINT = "facebook/mask2former-swin-tiny-ade-semantic"
WANDB_PROJECT = "smallDefectDetection"
WANDB_RUN_NAME = f"LiteJoint_stage{STAGE}"

STAGE1_CHECKPOINT_PATH = os.environ.get("STAGE1_CHECKPOINT_PATH", "")  # STAGE=2 warm-starts the det head from this

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMG_SIZE = 640
BATCH_SIZE = 4  # same Swin-Tiny + pixel-decoder memory footprint that already forced
                # DefectFormer and the Mask2Former baseline to batch=4 -- plus a stride-4
                # (P2) detection neck/head on top, which risk #4 in the plan flags as the
                # single biggest new memory cost. Stage 1 measures this for real before
                # Stage 2 commits to a full run at this batch size.
NUM_WORKERS = 2

if STAGE == 1:
    MAX_EPOCHS = 5
    PATIENCE = 5  # no early stopping within this short a run -- just complete all 5 epochs
    FREEZE_BACKBONE = True
    SEG_LOSS_WEIGHT = 0.0  # detection-only: cleanest isolation of "does the new head learn at all"
    BASE_LR = 1e-3  # higher than stage 2 -- only the new, randomly-initialized neck+head are
                     # training here (frozen backbone), so there's no pretrained-weight
                     # catastrophic-forgetting risk to protect against at this stage.
    BACKBONE_LR = 0.0  # unused, backbone is frozen
else:
    MAX_EPOCHS = 50  # reasoned from real step-count arithmetic (see the approved plan,
    PATIENCE = 15    # Section 4): matches both YOLOv8n-P2PAN's own real ~110k-step budget
                     # at batch=4, and the Mask2Former baseline's own real 50-epoch/patience-15
                     # budget on this exact dataset -- a defensible estimate, not a guarantee.
    FREEZE_BACKBONE = False
    SEG_LOSS_WEIGHT = 1.0
    BASE_LR = 1e-4
    BACKBONE_LR = 1e-5  # matches both DefectFormer's and the Mask2Former baseline's existing convention

DET_LOSS_WEIGHT = 1.0  # flat 1:1 to start -- unlike DefectFormer, detection and segmentation
                        # here only share the backbone, not a query representation, so there's
                        # no shared-query gradient-dominance concern to pre-compensate for.
                        # A placeholder pending real data, same as every loss weight in this
                        # project's history -- not over-tuned before Stage 1 even runs.

NECK_CH = 128
REG_MAX = 16
NUM_DET_CLASSES = 1  # single "defect" class, matching this project's nc=1 / single-class
                      # convention everywhere else (YOLO baselines, D-FINE-S). NOT DefectFormer's
                      # 0=background/1=defect 2-class convention -- TAL's classes are direct
                      # column indices into this head's own C-class output, so index 0 IS "defect"
                      # here, not background. Every GT box gets class 0.
TAL_TOPK = 10
STRIDES = (4, 8, 16, 32)

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/lite_joint_run"))
RUN_DIR = BASE_DIR / "runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

random.seed(SEED)

# ============================================================
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
     "numpy", "pandas", "pillow", "wandb", "transformers>=4.51.0,<4.52.0", "safetensors", "scipy", "torchvision"],
    check=True,
)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import Mask2FormerForUniversalSegmentation

from lite_joint_model import LiteJointModel, compute_detection_loss
from tal_assigner import TaskAlignedAssigner
from dfl_loss import DetectionHeadLoss
from detection_metrics import DetectionAPAccumulator

print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if not torch.cuda.is_available():
    sys.exit("CUDA is not available on this pod. Check `nvidia-smi`.")

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
        "model": MODEL_LABEL, "stage": STAGE, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "base_lr": BASE_LR, "backbone_lr": BACKBONE_LR,
        "seed": SEED, "freeze_backbone": FREEZE_BACKBONE, "det_loss_weight": DET_LOSS_WEIGHT,
        "seg_loss_weight": SEG_LOSS_WEIGHT, "neck_ch": NECK_CH, "reg_max": REG_MAX,
        "num_det_classes": NUM_DET_CLASSES, "tal_topk": TAL_TOPK, "strides": STRIDES,
        "stage1_checkpoint_path": STAGE1_CHECKPOINT_PATH or None,
    },
)

if not DATASET_ROOT.exists():
    sys.exit(f"DATASET_ROOT ({DATASET_ROOT}) doesn't exist -- scp processed_output there first.")

# ============================================================
# Dataset indexing + split -- copied near-verbatim from train_novelty_runpod.py, same seed/
# ratio/stratification convention as the YOLOv8n-P2PAN and YOLOv10n baseline notebooks, which
# is what makes the eventual comparison against baselines_small_defect_detection.pdf fair.
# ============================================================


def index_files(directory, suffixes):
    return {p.stem: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes}


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
        image_dir, mask_dir, label_dir = bucket_root / "images", bucket_root / "masks", bucket_root / "labels_yolo"

        mask_index = index_files(mask_dir, IMAGE_EXTS)
        label_index = index_files(label_dir, {".txt"})
        matched = bucket_missing = 0

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTS or image_path.name.startswith("._"):
                continue
            mask_path = next((mask_index[s] for s in candidate_stems(image_path.stem, "mask") if s in mask_index), None)
            label_path = next((label_index[s] for s in candidate_stems(image_path.stem, "box") if s in label_index), None)
            if mask_path is None or label_path is None:
                bucket_missing += 1
                continue
            samples.append({
                "image_path": image_path, "mask_path": mask_path, "label_path": label_path,
                "dataset": dataset_name, "size": size_bucket, "stratum": f"{dataset_name}_{size_bucket}",
            })
            matched += 1
        missing_count += bucket_missing
        print(f"{dataset_name}/{size_bucket}: {matched} matched, {bucket_missing} missing")

print("Total matched:", len(samples), "| missing:", missing_count)
if len(samples) != 12670:
    sys.exit(f"Expected 12,670 image-mask-label triplets, found {len(samples)}.")

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

assert len(train_samples) == 8858
assert len(val_samples) == 1892
assert len(test_samples) == 1920
print(f"Train: {len(train_samples)}  Validation: {len(val_samples)}  Test: {len(test_samples)}")


def load_binary_mask(mask_path, target_size):
    mask = Image.open(mask_path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def load_yolo_boxes(label_path):
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = line.split()
        boxes.append([float(cx), float(cy), float(w), float(h)])
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class JointDetSegDataset(Dataset):
    def __init__(self, split_samples):
        self.samples = split_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        mask = load_binary_mask(sample["mask_path"], (IMG_SIZE, IMG_SIZE))
        boxes_cxcywh_norm = load_yolo_boxes(sample["label_path"])

        pixel_values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "pixel_values": pixel_values,
            "mask": torch.from_numpy(mask.copy()).long(),
            "boxes": boxes_cxcywh_norm,
            "classes": torch.zeros(boxes_cxcywh_norm.shape[0], dtype=torch.long),  # single class, index 0 -- see NUM_DET_CLASSES note above
            "size": sample["size"],
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "classes": [b["classes"] for b in batch],
        "sizes": [b["size"] for b in batch],
    }


def make_loader(split_samples, shuffle=False):
    return DataLoader(
        JointDetSegDataset(split_samples), batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
        collate_fn=collate_fn,
    )


train_loader = make_loader(train_samples, shuffle=True)
val_loader = make_loader(val_samples)
print("Train batches:", len(train_loader), "Validation batches:", len(val_loader))


def boxes_norm_cxcywh_to_pixel_xyxy(boxes_norm):
    """[N,4] normalized cxcywh -> [N,4] absolute-pixel xyxy, the space TAL/DFL operate in."""
    if boxes_norm.numel() == 0:
        return boxes_norm.new_zeros((0, 4))
    b = boxes_norm * IMG_SIZE
    return torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], dim=-1)


def post_process_semantic_segmentation(class_queries_logits, masks_queries_logits, target_sizes):
    """Direct port of Mask2FormerImageProcessor.post_process_semantic_segmentation -- verified
    2026-08-21 to produce byte-identical output to the real HF processor on a real forward pass.
    Used here instead of DefectFormer's own (confirmed buggy: max-over-queries then an absolute
    threshold on class_prob*mask_prob) so this model's segmentation numbers are evaluated the
    same way the Mask2Former baseline's numbers already are -- an apples-to-apples comparison."""
    masks_queries_logits = F.interpolate(masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False)
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    out = []
    for idx in range(class_queries_logits.shape[0]):
        resized = F.interpolate(segmentation[idx].unsqueeze(0), size=target_sizes[idx], mode="bilinear", align_corners=False)
        out.append(resized[0].argmax(dim=0))
    return torch.stack(out)


# ============================================================
# Model
# ============================================================
print(f"Loading {BACKBONE_CHECKPOINT}...")
m2f = Mask2FormerForUniversalSegmentation.from_pretrained(BACKBONE_CHECKPOINT, num_labels=2, ignore_mismatched_sizes=True)
model = LiteJointModel(m2f, num_det_classes=NUM_DET_CLASSES, reg_max=REG_MAX, neck_ch=NECK_CH, strides=STRIDES).to(DEVICE)

if FREEZE_BACKBONE:
    for p in model.m2f.parameters():
        p.requires_grad = False
    print("Backbone frozen (Stage 1): only neck + detect head are trainable.")
    trainable_params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": BASE_LR}]
else:
    backbone_params = [p for n, p in model.named_parameters() if "m2f.model.pixel_level_module.encoder" in n]
    other_params = [p for n, p in model.named_parameters() if "m2f.model.pixel_level_module.encoder" not in n]
    trainable_params = [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": other_params, "lr": BASE_LR},
    ]
    if STAGE1_CHECKPOINT_PATH:
        init_path = Path(STAGE1_CHECKPOINT_PATH)
        if not init_path.exists():
            sys.exit(f"STAGE1_CHECKPOINT_PATH ({init_path}) doesn't exist -- scp Stage 1's checkpoint there first.")
        print(f"Warm-starting neck+head from {init_path} (Stage 1's frozen-backbone run)...")
        stage1_ckpt = torch.load(init_path, map_location=DEVICE, weights_only=False)
        load_result = model.load_state_dict(stage1_ckpt["model_state_dict"], strict=True)
        print(f"  Loaded cleanly (strict=True): backbone + neck + head all match Stage 1's checkpoint shapes.")

optimizer = AdamW(trainable_params, weight_decay=0.05)

assigner = TaskAlignedAssigner(topk=TAL_TOPK, alpha=1.0, beta=6.0)
head_loss_fn = DetectionHeadLoss(reg_max=REG_MAX)


@torch.inference_mode()
def evaluate(loader, measure_inference=False):
    model.eval()
    tp = fp = fn = 0
    ap_acc = DetectionAPAccumulator()
    inference_seconds = 0.0
    image_count = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        gt_mask = batch["mask"].to(DEVICE, non_blocking=True)

        if measure_inference:
            torch.cuda.synchronize()
            start = time.perf_counter()
        outputs = model(pixel_values)
        if measure_inference:
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start

        if SEG_LOSS_WEIGHT > 0:
            target_sizes = [(IMG_SIZE, IMG_SIZE)] * pixel_values.shape[0]
            pred_mask = post_process_semantic_segmentation(
                outputs["seg_class_logits"], outputs["seg_mask_logits"], target_sizes)
            tp += int(((pred_mask == 1) & (gt_mask == 1)).sum().item())
            fp += int(((pred_mask == 1) & (gt_mask == 0)).sum().item())
            fn += int(((pred_mask == 0) & (gt_mask == 1)).sum().item())

        scores, boxes = model.det_head.decode(outputs["det_cls_logits"], outputs["det_reg_logits"],
                                               outputs["anchor_points"], outputs["anchor_strides"])
        for b in range(pixel_values.shape[0]):
            gt_boxes_b = boxes_norm_cxcywh_to_pixel_xyxy(batch["boxes"][b]).to(DEVICE)
            gt_sizes_b = [batch["sizes"][b]] * gt_boxes_b.shape[0]
            ap_acc.add_image(boxes[b], scores[b, :, 0], gt_boxes_b, gt_sizes=gt_sizes_b)

        image_count += pixel_values.shape[0]

    seg_precision = tp / max(tp + fp, 1)
    seg_recall = tp / max(tp + fn, 1)
    seg_iou = tp / max(tp + fp + fn, 1)
    seg_dice = 2 * tp / max(2 * tp + fp + fn, 1)
    det_metrics = ap_acc.compute()

    return {
        "precision": seg_precision, "recall": seg_recall, "iou": seg_iou, "dice": seg_dice,
        "fp_per_image": fp / max(image_count, 1),
        "inference_time_ms_per_image": 1000 * inference_seconds / max(image_count, 1),
        "images": image_count,
        **det_metrics,
    }


# ============================================================
history = []
best_metric = -1.0
best_epoch = -1
epochs_without_improvement = 0
PROGRESS_EVERY = 50
peak_memory_logged = False

# Stage 2 tracks and saves TWO checkpoints, not one -- v4's actual run (see
# project_novelty_architecture.md) showed a real mAP50 peak (epoch 12) that Dice-only
# selection never saved because a different epoch (7) had better Dice. Repeating that same
# single-metric selection here would silently reintroduce the exact problem just diagnosed.
# Stage 1 only ever tracks mAP50 (no Dice signal, SEG_LOSS_WEIGHT=0), so this only matters
# for Stage 2.
best_dice = -1.0
best_dice_epoch = -1
best_map50 = -1.0
best_map50_epoch = -1

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    if FREEZE_BACKBONE:
        model.m2f.eval()  # frozen backbone stays in eval mode -- no BatchNorm/dropout drift on params that never update
    running_loss = running_det = running_seg = 0.0
    running_n_fg = 0
    epoch_start = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader, start=1):
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        gt_mask = batch["mask"].to(DEVICE, non_blocking=True)
        gt_boxes = [boxes_norm_cxcywh_to_pixel_xyxy(b).to(DEVICE, non_blocking=True) for b in batch["boxes"]]
        gt_classes = [c.to(DEVICE, non_blocking=True) for c in batch["classes"]]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(pixel_values)

        det_loss, det_parts = compute_detection_loss(
            outputs["det_cls_logits"], outputs["det_reg_logits"], outputs["anchor_points"], outputs["anchor_strides"],
            gt_boxes, gt_classes, assigner, head_loss_fn, num_classes=NUM_DET_CLASSES,
        )

        if SEG_LOSS_WEIGHT > 0:
            mask_labels = [gt_mask[i].float().unsqueeze(0) for i in range(gt_mask.shape[0])]
            class_labels = [torch.tensor([1], device=DEVICE) for _ in range(gt_mask.shape[0])]
            seg_loss_dict = model.m2f.criterion(
                masks_queries_logits=outputs["seg_mask_logits"], class_queries_logits=outputs["seg_class_logits"],
                mask_labels=mask_labels, class_labels=class_labels,
            )
            seg_loss = sum(seg_loss_dict.values())
        else:
            seg_loss = pixel_values.new_tensor(0.0)

        loss = DET_LOSS_WEIGHT * det_loss + SEG_LOSS_WEIGHT * seg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item())
        running_det += float(det_loss.item())
        running_seg += float(seg_loss.item())
        running_n_fg += det_parts["n_fg"]

        if not peak_memory_logged and batch_idx == 20:
            # Risk #4 from the plan: measure real peak memory before committing to a full run,
            # not after discovering an OOM partway through one.
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  [memory check @ batch 20] peak allocated so far: {peak_mb:.0f} MB "
                  f"(batch_size={BATCH_SIZE}, stage={STAGE})", flush=True)
            wandb.log({"peak_memory_mb_early": peak_mb}, step=epoch)
            peak_memory_logged = True

        if batch_idx % PROGRESS_EVERY == 0 or batch_idx == len(train_loader):
            elapsed = time.perf_counter() - epoch_start
            rate = elapsed / batch_idx
            eta = rate * (len(train_loader) - batch_idx)
            print(f"  epoch {epoch:02d} batch {batch_idx}/{len(train_loader)} "
                  f"| loss={running_loss/batch_idx:.4f} (det={running_det/batch_idx:.4f} "
                  f"seg={running_seg/batch_idx:.4f}) | avg_fg_anchors/batch={running_n_fg/batch_idx:.1f} "
                  f"| {rate:.2f}s/batch | ETA={eta/60:.1f}m", flush=True)

    val_metrics = evaluate(val_loader)
    row = {"epoch": epoch, "train_loss": running_loss / len(train_loader), **val_metrics}
    history.append(row)
    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={row['train_loss']:.4f} | val mAP50={row['mAP50']:.4f} "
          f"| val mAP50_Small={row['mAP50_Small']:.4f} | val Dice={row['dice']:.4f}", flush=True)
    wandb.log(row, step=epoch)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if STAGE == 1:
        # No Dice signal here (SEG_LOSS_WEIGHT=0) -- mAP50 is the only thing to select on,
        # and the only thing Stage 1 exists to measure.
        if row["mAP50"] > best_map50:
            best_map50 = row["mAP50"]
            best_map50_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_metrics},
                       RUN_DIR / "best_model.pt")
        else:
            epochs_without_improvement += 1
    else:
        # Two independent checkpoints, evaluated independently at the end -- see the comment
        # on best_dice/best_map50 above for why a single Dice-only selection isn't enough here.
        improved = False
        if row["dice"] > best_dice:
            best_dice, best_dice_epoch, improved = row["dice"], epoch, True
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_metrics},
                       RUN_DIR / "best_model_dice.pt")
        if row["mAP50"] > best_map50:
            best_map50, best_map50_epoch, improved = row["mAP50"], epoch, True
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_metrics},
                       RUN_DIR / "best_model_map50.pt")
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1

    pd.DataFrame(history).to_csv(RUN_DIR / "training_history.csv", index=False)
    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}; best Dice={best_dice:.4f}@{best_dice_epoch}, "
              f"best mAP50={best_map50:.4f}@{best_map50_epoch}.")
        break

wandb.summary["best_dice_epoch"] = best_dice_epoch
wandb.summary["best_dice"] = best_dice
wandb.summary["best_map50_epoch"] = best_map50_epoch
wandb.summary["best_map50"] = best_map50
print(f"Best Dice: {best_dice:.4f} at epoch {best_dice_epoch}. Best mAP50: {best_map50:.4f} at epoch {best_map50_epoch}.")

if STAGE == 1:
    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE. Do not proceed to Stage 2 automatically -- report these numbers")
    print("and get explicit go-ahead first, per this project's standing convention.")
    print(f"Best val mAP50: {best_map50:.4f} at epoch {best_map50_epoch}.")
    print(f"Checkpoint saved to: {RUN_DIR / 'best_model.pt'} (scp this down; STAGE=2 needs it")
    print(f"via STAGE1_CHECKPOINT_PATH if Stage 1 looks worth building on).")
    print("=" * 70)
    wandb.finish()
    sys.exit(0)

# ============================================================
# STAGE == 2 only past this point: full test-set evaluation against every real baseline.
# Both checkpoints get evaluated and reported -- see the best_dice/best_map50 comment above.
# ============================================================
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(RUN_DIR / "training_history.csv", FINAL_OUTPUT_DIR / "training_history.csv")

summary_rows = []
for selector, ckpt_name, sel_epoch in [("dice", "best_model_dice.pt", best_dice_epoch),
                                        ("map50", "best_model_map50.pt", best_map50_epoch)]:
    ckpt_path = RUN_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path} (never improved on this metric) -- skipping.")
        continue

    print(f"\n--- Test-set evaluation: {selector}-selected checkpoint (epoch {sel_epoch}) ---")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_rows = []
    for split_name, split_samples in test_sets.items():
        metrics = evaluate(make_loader(split_samples), measure_inference=True)
        test_rows.append({"split": split_name, **metrics})
        print(split_name, metrics)
        wandb.log({f"test_{selector}_{split_name}_{k}": v for k, v in metrics.items()})

    test_df = pd.DataFrame(test_rows)
    print(test_df.to_string(index=False))
    shutil.copy2(ckpt_path, FINAL_OUTPUT_DIR / ckpt_name)
    test_df.to_csv(FINAL_OUTPUT_DIR / f"evaluation_metrics_{selector}.csv", index=False)

    by_split = {row["split"]: row for row in test_rows}
    overall = by_split["overall"]
    summary_rows.append({
        "Experiment": f"{RUN_NAME}_{selector}selected", "Model": MODEL_LABEL, "Batch": BATCH_SIZE, "Epochs": sel_epoch,
        "mAP50": overall["mAP50"], "mAP50_95": overall["mAP50_95"],
        "Precision": overall["precision"], "Recall": overall["recall"],
        "mAP50_Small": by_split["small"]["mAP50"], "mAP50_Medium": by_split["medium"]["mAP50"], "mAP50_Large": by_split["large"]["mAP50"],
        "Recall_Small": by_split["small"]["recall"], "Recall_Medium": by_split["medium"]["recall"], "Recall_Large": by_split["large"]["recall"],
        "Inference_Time_ms": overall["inference_time_ms_per_image"], "FP_per_Image": overall["fp_per_image"],
        "Dice": overall["dice"], "IoU": overall["iou"],
        "Notes": f"{MODEL_LABEL}. Checkpoint selected by best val {selector} (epoch {sel_epoch}). "
                 f"Shared Swin-Tiny backbone (frozen in Stage 1, unfrozen w/ differential LR in Stage 2), "
                 f"TAL-assigned anchor-free (DFL) detect head on raw P2-P5 Swin stage features, "
                 f"Mask2Former's own unmodified segmentation path. Segmentation eval uses the same "
                 f"post_process_semantic_segmentation method as the Mask2Former baseline (verified "
                 f"byte-identical 2026-08-21), not DefectFormer's earlier, confirmed-buggy eval math.",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)
print("\nSaved final artifacts to:", FINAL_OUTPUT_DIR)
