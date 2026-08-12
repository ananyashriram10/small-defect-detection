"""
CrackDet (Chen et al., ICCV 2023, "The Devil is in the Crack Orientation") -- RunPod
training script, faithful from-spec reimplementation (see model/ + README.md).

This is a NICHE addition, not one of the project's official 8 detection / 8
segmentation baselines (see project memory: crack-specific methods were
explicitly cut from that list). It lives at baselines/niche/crackDet/. There
is no official CrackDet code release and no public copy of the paper's own
datasets (ONPP, ORC, OCCSD), so unlike every other script in this project,
this one has NO built-in dataset download step -- it expects the caller to
point it at oriented-box annotations already converted into this package's
JSON schema (see data/dataset.py's module docstring for the exact format).

Run:
    export WANDB_API_KEY=<your wandb key>              # optional, else interactive login
    export TRAIN_ANNOTATIONS=/workspace/data/train.json
    export VAL_ANNOTATIONS=/workspace/data/val.json
    export TEST_ANNOTATIONS=/workspace/data/test.json
    export IMAGE_ROOT=/workspace/data/images            # optional, defaults to annotation file's dir
    nohup python -u train_crackdet_runpod.py > train.log 2>&1 &
    tail -f train.log

Hyperparameters below are the paper's own (Sec. 4.1): Adam, lr 4e-4 decayed
by 10x at epochs 20/40, 60 total epochs, batch 32, 512x512 input,
lambda_off=0.1 / lambda_size=0.2 / lambda_mar=0.1 (matching the ONPP/ORC/
OCCSD training recipe -- the closest of the paper's 4 datasets to this
project's likely image sizes; adjust EPOCHS/LR_DECAY_EPOCHS if you point
this at HRSC2016-scale data instead, see Sec. 4.1's separate 140-epoch
recipe for that dataset).
"""

import os

# Defensive, same reasoning as every other RunPod script in this project.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# Config -- paper hyperparameters (Sec. 4.1), ONPP/ORC/OCCSD recipe.
# ============================================================
RUN_NAME = "RunCrackDet_oriented_subcrack"
MODEL_LABEL = "CrackDet"
WANDB_PROJECT = "smallDefectDetection"
WANDB_RUN_NAME = f"{MODEL_LABEL}_oriented_detection"

SEED = 42
INPUT_SIZE = 512
STRIDE = 4
NUM_CLASSES = 1
GROUP_ORDER = 8  # rotation group C_8 for the e2cnn backbone -- see model/backbone.py docstring

BATCH_SIZE = 32
MAX_EPOCHS = 60
LR = 4e-4
LR_DECAY_EPOCHS = [20, 40]
LR_DECAY_FACTOR = 0.1
LAMBDA_OFF = 0.1
LAMBDA_SIZE = 0.2
LAMBDA_MAR = 0.1
NUM_WORKERS = 4

SCORE_THRESH = 0.3           # for precision/recall/MOE eval
NMS_TOPK = 200
MATCH_IOU_THRESH = 0.5       # standard oriented-detection matching threshold (DOTA convention)

TRAIN_ANNOTATIONS = os.environ.get("TRAIN_ANNOTATIONS")
VAL_ANNOTATIONS = os.environ.get("VAL_ANNOTATIONS")
TEST_ANNOTATIONS = os.environ.get("TEST_ANNOTATIONS")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT")  # optional; defaults to each annotation file's own dir

BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/crackdet_run"))
RUN_DIR = BASE_DIR / "runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

if not TRAIN_ANNOTATIONS or not VAL_ANNOTATIONS or not TEST_ANNOTATIONS:
    sys.exit(
        "TRAIN_ANNOTATIONS / VAL_ANNOTATIONS / TEST_ANNOTATIONS must all be set -- there is no "
        "auto-downloaded dataset for CrackDet (the paper's ONPP/ORC/OCCSD datasets were never "
        "released publicly, and this repo has no oriented sub-crack box annotations of its own "
        "yet). Point these at JSON files following the schema documented in "
        "baselines/niche/crackDet/data/dataset.py, then re-run.\n"
        "  export TRAIN_ANNOTATIONS=/workspace/data/train.json\n"
        "  export VAL_ANNOTATIONS=/workspace/data/val.json\n"
        "  export TEST_ANNOTATIONS=/workspace/data/test.json"
    )

# ============================================================
# e2cnn/wandb/shapely are not part of a stock RunPod PyTorch image.
# ============================================================
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
     "e2cnn", "wandb", "pandas", "pillow", "numpy", "shapely"],
    check=True,
)

import numpy as np
import pandas as pd
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import OrientedCrackDataset, crackdet_collate_fn
from model.crackdet import CrackDet
from model.losses import CrackDetLoss
from model.piecewise_angle import decode_box
from model.postprocess import decode as decode_predictions

print("torch:", torch.__version__, "| torch's CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if not torch.cuda.is_available():
    sys.exit("CUDA is not available on this pod. Check the GPU is attached and the driver/CUDA toolkit is loaded (`nvidia-smi`).")

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
        "model": MODEL_LABEL,
        "backbone": "ReEDNet (e2cnn ResNet-50 layout, C_8 group)",
        "input_size": INPUT_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "lr": LR,
        "lr_decay_epochs": LR_DECAY_EPOCHS,
        "lambda_off": LAMBDA_OFF,
        "lambda_size": LAMBDA_SIZE,
        "lambda_mar": LAMBDA_MAR,
        "seed": SEED,
    },
)

# ============================================================
# Data
# ============================================================
train_ds = OrientedCrackDataset(TRAIN_ANNOTATIONS, image_root=IMAGE_ROOT, input_size=INPUT_SIZE,
                                 stride=STRIDE, num_classes=NUM_CLASSES, augment=True)
val_ds = OrientedCrackDataset(VAL_ANNOTATIONS, image_root=IMAGE_ROOT, input_size=INPUT_SIZE,
                               stride=STRIDE, num_classes=NUM_CLASSES, augment=False)
test_ds = OrientedCrackDataset(TEST_ANNOTATIONS, image_root=IMAGE_ROOT, input_size=INPUT_SIZE,
                                stride=STRIDE, num_classes=NUM_CLASSES, augment=False)

print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images | Test: {len(test_ds)} images")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                           pin_memory=True, collate_fn=crackdet_collate_fn,
                           persistent_workers=NUM_WORKERS > 0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True, collate_fn=crackdet_collate_fn,
                         persistent_workers=NUM_WORKERS > 0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                          pin_memory=True, collate_fn=crackdet_collate_fn,
                          persistent_workers=NUM_WORKERS > 0)

# ============================================================
# Model / loss / optimizer
# ============================================================
model = CrackDet(num_classes=NUM_CLASSES, group_order=GROUP_ORDER).to(DEVICE)
criterion = CrackDetLoss(lambda_off=LAMBDA_OFF, lambda_size=LAMBDA_SIZE, lambda_mar=LAMBDA_MAR)
optimizer = Adam(model.parameters(), lr=LR)
scheduler = MultiStepLR(optimizer, milestones=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"CrackDet parameters: {n_params:,} total, {n_trainable:,} trainable")
wandb.summary["param_count"] = n_params

RUN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Rotated-box IoU matching for evaluation (Precision/Recall/MOE, Sec. 4.1).
# ============================================================
def _polygon(cx, cy, h, w, theta_deg):
    from shapely.geometry import Polygon
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    local = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
    rot = np.stack([local[:, 0] * cos_t - local[:, 1] * sin_t,
                     local[:, 0] * sin_t + local[:, 1] * cos_t], axis=1)
    rot[:, 0] += cx
    rot[:, 1] += cy
    return Polygon(rot).buffer(0)


def _iou(box_a, box_b) -> float:
    pa, pb = _polygon(*box_a), _polygon(*box_b)
    if not pa.is_valid or not pb.is_valid or pa.area == 0 or pb.area == 0:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return inter / union if union > 0 else 0.0


def _angle_error_rad(theta_pred_deg: float, theta_gt_deg: float) -> float:
    diff = abs(theta_pred_deg - theta_gt_deg) % 180.0
    diff = min(diff, 180.0 - diff)  # a line has 180-degree periodicity, not 360
    return np.deg2rad(diff)


@torch.no_grad()
def evaluate(loader, iou_thresh=MATCH_IOU_THRESH, score_thresh=SCORE_THRESH):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    tp = fp = fn = 0
    angle_errors = []
    inference_seconds = 0.0
    n_images = 0

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = model(images)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - t0
        n_images += images.shape[0]

        loss_dict = criterion(preds, targets)
        total_loss += float(loss_dict["loss"].item())
        n_batches += 1

        decoded = decode_predictions(preds, topk=NMS_TOPK, score_thresh=score_thresh,
                                      stride=STRIDE)
        gt_hm = targets["heatmap"].shape
        for b in range(images.shape[0]):
            pred_boxes = decoded[b]["boxes"].cpu().numpy()
            pred_scores = decoded[b]["scores"].cpu().numpy()
            gt_mask = (targets["batch_idx"] == b)
            gt_pix = targets["pixel_idx"][gt_mask].cpu().numpy()
            gt_off = targets["offset"][gt_mask].cpu().numpy()
            gt_size = targets["size"][gt_mask].cpu().numpy()
            gt_theta_i = targets["theta_i"][gt_mask].cpu().numpy()
            gt_branch = targets["branch_idx"][gt_mask].cpu().numpy()

            fw = gt_hm[-1]
            gt_boxes = []
            for k in range(len(gt_pix)):
                iy, ix = divmod(int(gt_pix[k]), fw)
                cx = (ix + gt_off[k, 0]) * STRIDE
                cy = (iy + gt_off[k, 1]) * STRIDE
                from model.piecewise_angle import decode_box
                h_t = torch.tensor([gt_size[k, 0]])
                w_t = torch.tensor([gt_size[k, 1]])
                th_t = torch.tensor([gt_theta_i[k]])
                br_t = torch.tensor([int(gt_branch[k])])
                h_o, w_o, theta_o = decode_box(h_t, w_t, th_t, br_t)
                gt_boxes.append((cx, cy, float(h_o.item()) * STRIDE, float(w_o.item()) * STRIDE,
                                  float(theta_o.item())))

            order = np.argsort(-pred_scores)
            matched_gt = set()
            for idx in order:
                best_iou, best_gt = 0.0, -1
                for gi, gt_box in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = _iou(tuple(pred_boxes[idx]), gt_box)
                    if iou > best_iou:
                        best_iou, best_gt = iou, gi
                if best_iou >= iou_thresh:
                    tp += 1
                    matched_gt.add(best_gt)
                    angle_errors.append(_angle_error_rad(float(pred_boxes[idx][4]), gt_boxes[best_gt][4]))
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    moe = float(np.mean(angle_errors)) if angle_errors else float("nan")
    return {
        "loss": total_loss / max(n_batches, 1),
        "precision": precision,
        "recall": recall,
        "moe": moe,
        "tp": tp, "fp": fp, "fn": fn,
        "inference_time_ms_per_image": 1000 * inference_seconds / max(n_images, 1),
    }


# ============================================================
# Training loop, with resume support (same pattern as every other RunPod script here).
# ============================================================
history = []
best_recall = -1.0
best_epoch = -1
start_epoch = 1
PROGRESS_EVERY = 50

history_path = RUN_DIR / "training_history.csv"
checkpoint_path = RUN_DIR / "best_model.pt"
if history_path.exists() and checkpoint_path.exists():
    print(f"Found existing checkpoint + history at {RUN_DIR} -- resuming instead of restarting.")
    resume = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(resume["model_state_dict"])
    optimizer.load_state_dict(resume["optimizer_state_dict"])
    scheduler.load_state_dict(resume["scheduler_state_dict"])
    best_epoch = resume["epoch"]
    best_recall = resume["val_metrics"]["recall"]
    history_df = pd.read_csv(history_path)
    history = history_df.to_dict("records")
    start_epoch = int(history_df["epoch"].max()) + 1
    print(f"Resuming from epoch {start_epoch}. Best so far: epoch {best_epoch}, val recall={best_recall:.4f}.")
else:
    print("No existing checkpoint found -- starting fresh.")

for epoch in range(start_epoch, MAX_EPOCHS + 1):
    model.train()
    running = {"loss": 0.0, "loss_heatmap": 0.0, "loss_offset": 0.0, "loss_size": 0.0, "loss_mar": 0.0}
    epoch_start = time.perf_counter()

    for batch_idx, (images, targets) in enumerate(train_loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss_dict = criterion(preds, targets)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        optimizer.step()

        for k in running:
            running[k] += float(loss_dict[k].item())

        if batch_idx % PROGRESS_EVERY == 0 or batch_idx == len(train_loader):
            elapsed = time.perf_counter() - epoch_start
            rate = elapsed / batch_idx
            eta = rate * (len(train_loader) - batch_idx)
            print(f"  epoch {epoch:02d} batch {batch_idx}/{len(train_loader)} "
                  f"| avg loss={running['loss'] / batch_idx:.4f} "
                  f"| {rate:.2f}s/batch | ETA this epoch={eta/60:.1f}m", flush=True)

    scheduler.step()
    val_metrics = evaluate(val_loader)
    row = {"epoch": epoch, **{f"train_{k}": v / len(train_loader) for k, v in running.items()},
           **{f"val_{k}": v for k, v in val_metrics.items()},
           "lr": optimizer.param_groups[0]["lr"]}
    history.append(row)
    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | train_loss={row['train_loss']:.4f} "
          f"| val_precision={row['val_precision']:.4f} | val_recall={row['val_recall']:.4f} "
          f"| val_MOE={row['val_moe']:.4f}", flush=True)
    wandb.log(row, step=epoch)

    if val_metrics["recall"] > best_recall:
        best_recall = val_metrics["recall"]
        best_epoch = epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
        }, checkpoint_path)

    pd.DataFrame(history).to_csv(history_path, index=False)

wandb.summary["best_epoch"] = best_epoch
wandb.summary["best_validation_recall"] = best_recall
print("Best epoch:", best_epoch, "Best validation recall:", best_recall)

# ============================================================
# Final test-set evaluation with the best checkpoint.
# ============================================================
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate(test_loader)
print("Test metrics:", test_metrics)
wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(checkpoint_path, FINAL_OUTPUT_DIR / "best_model.pt")
shutil.copy2(history_path, FINAL_OUTPUT_DIR / "training_history.csv")

summary_row = {
    "Experiment": RUN_NAME,
    "Model": MODEL_LABEL,
    "Batch": BATCH_SIZE,
    "Epochs": best_epoch,
    "Precision": test_metrics["precision"],
    "Recall": test_metrics["recall"],
    "MOE": test_metrics["moe"],
    "Inference_Time_ms": test_metrics["inference_time_ms_per_image"],
    "Notes": "Oriented sub-crack detector, faithfully reimplemented from Chen et al. ICCV 2023 "
             "(no official code release). Metrics are Precision/Recall (rotated-box IoU>=0.5 "
             "matching) and MOE (mean orientation error, radians, on matched boxes only), "
             "matching the paper's own Table 1/2 metrics -- NOT the detection/segmentation "
             "8-baseline summary.csv schema, since oriented sub-crack detection is a different "
             "task shape (see baselines/niche/crackDet/README.md).",
}
pd.DataFrame([summary_row]).to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)

metadata = {
    "experiment": RUN_NAME, "model": MODEL_LABEL, "task": "oriented sub-crack detection",
    "input_size": INPUT_SIZE, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
    "best_epoch": best_epoch, "best_validation_recall": best_recall,
    "param_count": n_params,
    "dataset": {"train": TRAIN_ANNOTATIONS, "val": VAL_ANNOTATIONS, "test": TEST_ANNOTATIONS},
}
(FINAL_OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

print("Saved final artifacts to:", FINAL_OUTPUT_DIR)
wandb.finish()
print("DONE.")
