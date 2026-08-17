"""
Novelty model: Mask2Former backbone + parallel detection queries + bidirectional
cross-attention between det/seg queries + training-only prompt-alignment auxiliary
task. RunPod version -- same scaffolding pattern as train_mask2former_runpod.py
(same dataset structure, same download-if-missing fallback, same resume-on-restart
checkpointing), since this model is at least as heavy as plain Mask2Former was
(same backbone, plus a second Hungarian matcher for detection on top of
Mask2Former's own for segmentation). AMP is enabled (bf16 autocast around
forward passes only, loss/matching math forced to fp32) -- see the training
loop comment for why, given D-FINE's own AMP run on this dataset NaN'd.

Run:
    export WANDB_API_KEY=<your wandb key>
    export DATASET_ROOT=/workspace/dataset          # scp'd + untarred processed_output
    export PROMPTS_JSON=/workspace/defect_prompts_v2.json
    nohup python -u train_novelty_runpod.py > train.log 2>&1 &
    tail -f train.log

novelty_experiments/*.py must sit next to this script on the pod (scp the whole
novelty_experiments/ folder alongside the dataset and prompts).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# novelty_experiments/*.py needs to be importable -- scp it next to this script.
sys.path.insert(0, str(Path(__file__).parent / "novelty_experiments"))

# ============================================================
# Config
# ============================================================
RUN_NAME = "RunP_novelty_m2f_crossattn_imgsz640"
MODEL_LABEL = "Novelty (Mask2Former + cross-attn + prompt-alignment)"
BACKBONE_CHECKPOINT = "facebook/mask2former-swin-tiny-ade-semantic"
WANDB_PROJECT = "smallDefectDetection"
WANDB_RUN_NAME = "Novelty_joint_det_seg"

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMG_SIZE = 640
BATCH_SIZE = 4  # migrated from a 16GB A4000 (batch=2, fp32, 14,091/16,376 MiB at batch=1
                # -- see git history/memory for that measurement) to a 24GB A5000, plus AMP
                # (bf16 forward passes) now landing at the same time -- both raise the real
                # ceiling, but this exact combination was never measured, so it's a real bet,
                # not a confirmed-safe value. Check nvidia-smi a few minutes in, specifically
                # on a batch where the alignment task fires (every ALIGN_EVERY_N_BATCHES-th),
                # since that's still the actual peak-memory moment. If it OOMs, drop to 3 or
                # 2; the resume checkpoint means that costs a restart, not lost progress.
MAX_EPOCHS = 50
PATIENCE = 15
BASE_LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 0.05
NUM_WORKERS = 2

# Relative loss weights. Detection and segmentation losses are each already
# internally calibrated by their own established recipes (DETR-style class/bbox/giou
# weights inside DetectionLoss; Mask2Former's own mask/dice/CE weights inside its
# criterion), so they're combined at native scale (1:1). The alignment loss is meant
# to be a light regularizer, not compete with the main task losses, so it's downweighted
# and only run periodically -- both are placeholder choices, not tuned, flag if you want
# to revisit either.
DET_LOSS_WEIGHT = 1.0
SEG_LOSS_WEIGHT = 1.0
ALIGN_LOSS_WEIGHT = 0.1
ALIGN_EVERY_N_BATCHES = 5
ALIGN_BATCH_SIZE = 4  # dropped from 16 -- this is the actual OOM trigger: it runs through
                      # the same Swin backbone as the main batch, but while the main
                      # batch's activations are still held in memory for the later
                      # combined backward pass, so it stacks on top rather than replacing.
                      # An auxiliary contrastive task doesn't need a big batch to work.

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))
PROMPTS_JSON = Path(os.environ.get("PROMPTS_JSON", "/workspace/defect_prompts_v2.json"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/novelty_run"))
RUN_DIR = BASE_DIR / "novelty_runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

random.seed(SEED)

# ============================================================
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
     "numpy", "pandas", "pillow", "wandb", "transformers>=4.51.0,<4.52.0", "safetensors", "scipy"],
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

from novelty_model import NoveltyModel
from detection_loss import DetectionLoss, box_cxcywh_to_xyxy
from detection_metrics import DetectionAPAccumulator
from prompt_pair_dataset import DefectPromptPairDataset, resolve_image_path as resolve_prompt_image_path
from prompt_alignment_head import PromptAlignmentHead, PromptVocab

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
        "model": MODEL_LABEL, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "base_lr": BASE_LR,
        "backbone_lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY, "seed": SEED,
        "det_loss_weight": DET_LOSS_WEIGHT, "seg_loss_weight": SEG_LOSS_WEIGHT,
        "align_loss_weight": ALIGN_LOSS_WEIGHT, "align_every_n_batches": ALIGN_EVERY_N_BATCHES,
        "amp": True, "amp_dtype": "bfloat16",
    },
)

if not DATASET_ROOT.exists():
    sys.exit(f"DATASET_ROOT ({DATASET_ROOT}) doesn't exist -- scp processed_output there first.")
if not PROMPTS_JSON.exists():
    sys.exit(f"PROMPTS_JSON ({PROMPTS_JSON}) doesn't exist -- scp defect_prompts_v2.json there first.")

# ============================================================
# Index images + masks + YOLO boxes (same matching logic as every other script here,
# extended to also load the box label file, not just the mask).
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
    mask = Image.open(mask_path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def load_yolo_boxes(label_path):
    """YOLO format: 'class cx cy w h', already normalized 0-1 -- matches DetectionLoss's
    expected cxcywh format directly, no conversion needed. Binary task: every real box
    gets class 1 (0 reserved for background/no-object), matching every other baseline's
    'defect' vs 'background' convention in this project."""
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
        boxes = load_yolo_boxes(sample["label_path"])

        pixel_values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "pixel_values": pixel_values,
            "mask": torch.from_numpy(mask.copy()).long(),
            "boxes": boxes,
            "classes": torch.ones(boxes.shape[0], dtype=torch.long),
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

# ============================================================
# Auxiliary prompt-alignment dataset -- separate loader, cycled through periodically,
# not every main-loop step (see ALIGN_EVERY_N_BATCHES).
# ============================================================
print("Building prompt vocabulary from the real corpus...")
vocab = PromptVocab.build_from_prompts_json(str(PROMPTS_JSON))
print(f"Vocab size: {len(vocab)}")

align_dataset = DefectPromptPairDataset(prompts_json_path=str(PROMPTS_JSON), dataset_root=str(DATASET_ROOT), seed=SEED)
print(f"Alignment dataset: {len(align_dataset)} instances, {align_dataset.skipped} skipped.")


def align_collate_fn(batch):
    crops = torch.stack([
        torch.from_numpy(np.asarray(item["image_crop"].resize((IMG_SIZE, IMG_SIZE))).copy()).permute(2, 0, 1).float() / 255.0
        for item in batch
    ])
    token_ids = torch.stack([vocab.encode(item["prompt_text"]) for item in batch])
    return crops, token_ids


align_loader = DataLoader(
    align_dataset, batch_size=ALIGN_BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    collate_fn=align_collate_fn,
)


def cycle(loader):
    while True:
        for item in loader:
            yield item


align_iter = cycle(align_loader)

# ============================================================
# Model
# ============================================================
m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
    BACKBONE_CHECKPOINT, num_labels=2, ignore_mismatched_sizes=True,
)
model = NoveltyModel(m2f, num_classes=2, num_det_queries=100).to(DEVICE)
align_head = PromptAlignmentHead(m2f, vocab_size=len(vocab)).to(DEVICE)

det_loss_fn = DetectionLoss(num_classes=2).to(DEVICE)
seg_criterion = m2f.criterion

backbone_params = [p for n, p in model.named_parameters() if "m2f.model.pixel_level_module.encoder" in n]
other_params = [p for n, p in model.named_parameters() if "m2f.model.pixel_level_module.encoder" not in n]
align_only_params = [p for n, p in align_head.named_parameters() if "image_encoder" not in n]  # exclude the shared backbone, already in `other_params`/`backbone_params`

optimizer = AdamW([
    {"params": backbone_params, "lr": BACKBONE_LR},
    {"params": other_params, "lr": BASE_LR},
    {"params": align_only_params, "lr": BASE_LR},
], weight_decay=WEIGHT_DECAY)

RUN_DIR.mkdir(parents=True, exist_ok=True)
n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_align_params = sum(p.numel() for p in align_only_params)
print(f"Trainable params: model={n_model_params:,}, alignment-head-only={n_align_params:,}")


def masks_to_boxes_xyxy(pred_mask_binary):
    """Fallback for turning a predicted binary mask into one enclosing box per connected
    blob isn't implemented here (would need scipy.ndimage.label); detection uses the
    dedicated detection head's own boxes directly, this helper is unused at present and
    kept out to avoid dead code -- boxes always come from det_boxes, not from seg masks."""
    raise NotImplementedError


@torch.inference_mode()
def evaluate(loader, measure_inference=False, score_threshold=0.5):
    model.eval()
    tp = fp = fn = 0  # segmentation pixel counts
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

        seg_pred = outputs["seg_mask_logits"].argmax(dim=1)  # background vs defect over the 2 real classes (query-agnostic collapse below)
        # seg_mask_logits is per-query [B,Q,H,W]; the actual semantic map is derived the
        # same way Mask2Former's own post-processing does: take the argmax class per query
        # via seg_class_logits, then compose a semantic map from the highest-scoring query
        # per pixel. Simplified here to match this project's binary-mask evaluation
        # convention (foreground = any query confidently predicting the defect class).
        class_probs = outputs["seg_class_logits"].softmax(-1)[..., 1]  # [B,Q] prob of "defect" class per query
        mask_probs = outputs["seg_mask_logits"].sigmoid()  # [B,Q,H,W]
        weighted = (class_probs.unsqueeze(-1).unsqueeze(-1) * mask_probs).amax(dim=1)  # [B,H,W]
        weighted = F.interpolate(weighted.unsqueeze(1), size=gt_mask.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        pred_mask = (weighted > 0.5).long()

        tp += int(((pred_mask == 1) & (gt_mask == 1)).sum().item())
        fp += int(((pred_mask == 1) & (gt_mask == 0)).sum().item())
        fn += int(((pred_mask == 0) & (gt_mask == 1)).sum().item())

        det_class_probs = outputs["det_class_logits"].softmax(-1)  # [B,Q,3]
        det_scores = det_class_probs[..., 1]  # prob of the single real class
        det_boxes_xyxy_all = box_cxcywh_to_xyxy(outputs["det_boxes"])

        for b in range(pixel_values.shape[0]):
            keep = det_scores[b] > score_threshold
            pred_boxes_b = det_boxes_xyxy_all[b][keep]
            pred_scores_b = det_scores[b][keep]
            gt_boxes_b = box_cxcywh_to_xyxy(batch["boxes"][b].to(DEVICE)) if batch["boxes"][b].numel() else torch.zeros((0, 4), device=DEVICE)
            gt_sizes_b = [batch["sizes"][b]] * gt_boxes_b.shape[0]
            ap_acc.add_image(pred_boxes_b, pred_scores_b, gt_boxes_b, gt_sizes=gt_sizes_b)

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
# Resume support -- same pattern as train_mask2former_runpod.py, extended to also
# save/restore the alignment head's own (non-shared-backbone) parameters.
# ============================================================
history = []
best_dice = -1.0
best_epoch = -1
epochs_without_improvement = 0
start_epoch = 1
PROGRESS_EVERY = 50

history_path = RUN_DIR / "training_history.csv"
checkpoint_path = RUN_DIR / "best_model.pt"
if history_path.exists() and checkpoint_path.exists():
    print(f"Found existing checkpoint + history at {RUN_DIR} -- resuming instead of restarting.")
    resume_checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(resume_checkpoint["model_state_dict"])
    align_head.load_state_dict(resume_checkpoint["align_head_state_dict"])
    optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    best_epoch = resume_checkpoint["epoch"]
    best_dice = resume_checkpoint["val_metrics"]["dice"]

    history_df = pd.read_csv(history_path)
    history = history_df.to_dict("records")
    last_logged_epoch = int(history_df["epoch"].max())
    start_epoch = last_logged_epoch + 1
    epochs_without_improvement = last_logged_epoch - best_epoch
    print(f"Resuming from epoch {start_epoch}. Best so far: epoch {best_epoch}, val Dice={best_dice:.4f}.")
else:
    print("No existing checkpoint found -- starting fresh from the pretrained Mask2Former backbone.")

for epoch in range(start_epoch, MAX_EPOCHS + 1):
    model.train()
    align_head.train()
    running_loss = running_det = running_seg = running_align = 0.0
    epoch_start = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader, start=1):
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        gt_mask = batch["mask"].to(DEVICE, non_blocking=True)
        gt_boxes = [b.to(DEVICE, non_blocking=True) for b in batch["boxes"]]
        gt_classes = [c.to(DEVICE, non_blocking=True) for c in batch["classes"]]

        optimizer.zero_grad(set_to_none=True)

        # bf16 (not fp16) specifically because fp16's narrow exponent range is the likely
        # culprit behind D-FINE's earlier NaN-box AMP failure on this same dataset -- bf16
        # has fp32's exponent range, just less mantissa precision, so no GradScaler needed.
        # Only forward passes run under autocast; det_loss_fn's GIoU + Hungarian matching
        # and seg_criterion's own mask Hungarian matcher are forced back to fp32 below
        # (explicit .float() casts) since that box/mask-area arithmetic is exactly what's
        # fragile under reduced precision -- not assumed safe just because it's untested here.
        do_align = batch_idx % ALIGN_EVERY_N_BATCHES == 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values)
            if do_align:
                align_crops, align_tokens = next(align_iter)
                align_crops = align_crops.to(DEVICE, non_blocking=True)
                align_tokens = align_tokens.to(DEVICE, non_blocking=True)
                align_loss, _ = align_head(align_crops, align_tokens)

        det_targets = [{"classes": gt_classes[i], "boxes": gt_boxes[i]} for i in range(len(gt_boxes))]
        det_loss, _ = det_loss_fn(
            outputs["det_class_logits"].float(), outputs["det_boxes"].float(), det_targets
        )

        mask_labels = [gt_mask[i].float().unsqueeze(0) for i in range(gt_mask.shape[0])]
        class_labels = [torch.tensor([1], device=DEVICE) for _ in range(gt_mask.shape[0])]
        seg_loss_dict = seg_criterion(
            masks_queries_logits=outputs["seg_mask_logits"].float(),
            class_queries_logits=outputs["seg_class_logits"].float(),
            mask_labels=mask_labels, class_labels=class_labels,
        )
        seg_loss = sum(seg_loss_dict.values())

        loss = DET_LOSS_WEIGHT * det_loss + SEG_LOSS_WEIGHT * seg_loss

        align_loss_value = 0.0
        if do_align:
            loss = loss + ALIGN_LOSS_WEIGHT * align_loss.float()
            align_loss_value = float(align_loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(align_head.parameters()), max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item())
        running_det += float(det_loss.item())
        running_seg += float(seg_loss.item())
        running_align += align_loss_value

        if batch_idx % PROGRESS_EVERY == 0 or batch_idx == len(train_loader):
            elapsed = time.perf_counter() - epoch_start
            rate = elapsed / batch_idx
            eta = rate * (len(train_loader) - batch_idx)
            print(f"  epoch {epoch:02d} batch {batch_idx}/{len(train_loader)} "
                  f"| loss={running_loss/batch_idx:.4f} (det={running_det/batch_idx:.4f} "
                  f"seg={running_seg/batch_idx:.4f} align={running_align/batch_idx:.4f}) "
                  f"| {rate:.2f}s/batch | ETA={eta/60:.1f}m", flush=True)

    val_metrics = evaluate(val_loader)
    row = {"epoch": epoch, "train_loss": running_loss / len(train_loader), **val_metrics}
    history.append(row)
    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={row['train_loss']:.4f} | val Dice={row['dice']:.4f} "
          f"| val mAP50={row['mAP50']:.4f} | val Recall={row['recall']:.4f}", flush=True)
    wandb.log(row, step=epoch)

    if row["dice"] > best_dice:
        best_dice = row["dice"]
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "align_head_state_dict": align_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, RUN_DIR / "best_model.pt")
    else:
        epochs_without_improvement += 1

    pd.DataFrame(history).to_csv(RUN_DIR / "training_history.csv", index=False)
    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}; best val Dice was {best_dice:.4f} at epoch {best_epoch}.")
        break

wandb.summary["best_epoch"] = best_epoch
wandb.summary["best_validation_dice"] = best_dice
print("Best epoch:", best_epoch, "Best validation Dice:", best_dice)

# ============================================================
checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

test_rows = []
for split_name, split_samples in test_sets.items():
    metrics = evaluate(make_loader(split_samples), measure_inference=True)
    test_rows.append({"split": split_name, **metrics})
    print(split_name, metrics)
    wandb.log({f"test_{split_name}_{k}": v for k, v in metrics.items()})

test_df = pd.DataFrame(test_rows)
print(test_df.to_string(index=False))

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(RUN_DIR / "best_model.pt", FINAL_OUTPUT_DIR / "best_model.pt")
shutil.copy2(RUN_DIR / "training_history.csv", FINAL_OUTPUT_DIR / "training_history.csv")
test_df.to_csv(FINAL_OUTPUT_DIR / "evaluation_metrics.csv", index=False)

by_split = {row["split"]: row for row in test_rows}
overall = by_split["overall"]

summary_row = {
    "Experiment": RUN_NAME, "Model": MODEL_LABEL, "Batch": BATCH_SIZE, "Epochs": best_epoch,
    "mAP50": overall["mAP50"], "mAP50_95": overall["mAP50_95"],
    "Precision": overall["precision"], "Recall": overall["recall"],
    "mAP50_Small": by_split["small"]["mAP50"], "mAP50_Medium": by_split["medium"]["mAP50"], "mAP50_Large": by_split["large"]["mAP50"],
    "Recall_Small": by_split["small"]["recall"], "Recall_Medium": by_split["medium"]["recall"], "Recall_Large": by_split["large"]["recall"],
    "Inference_Time_ms": overall["inference_time_ms_per_image"], "FP_per_Image": overall["fp_per_image"],
    "Dice": overall["dice"], "IoU": overall["iou"],
    "Notes": f"{MODEL_LABEL} on RunPod, joint det+seg with cross-attention, training-only prompt-alignment "
             f"auxiliary task (weight={ALIGN_LOSS_WEIGHT}, every {ALIGN_EVERY_N_BATCHES} batches), full precision "
             f"(no AMP), fixed 640 split. Both mAP (detection) and Dice/IoU (segmentation) reported since this "
             f"model does both tasks jointly, unlike the single-task baselines.",
}
summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)

metadata = {
    "experiment": RUN_NAME, "model": MODEL_LABEL, "task": "joint detection + segmentation",
    "image_size": IMG_SIZE, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE, "best_epoch": best_epoch, "best_validation_dice": best_dice,
    "det_loss_weight": DET_LOSS_WEIGHT, "seg_loss_weight": SEG_LOSS_WEIGHT,
    "align_loss_weight": ALIGN_LOSS_WEIGHT, "align_every_n_batches": ALIGN_EVERY_N_BATCHES,
    "split_counts": {
        "train": len(train_samples), "val": len(val_samples), "test": len(test_samples),
        "test_small": len(test_sets["small"]), "test_medium": len(test_sets["medium"]), "test_large": len(test_sets["large"]),
    },
}
(FINAL_OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

print("Saved final artifacts to:", FINAL_OUTPUT_DIR)
print(summary_df.to_string(index=False))
wandb.finish()
print("DONE.")
