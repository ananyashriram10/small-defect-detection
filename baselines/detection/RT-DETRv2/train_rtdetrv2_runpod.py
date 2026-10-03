"""
RT-DETRv2 (R18VD), 640 px, one-class small-defect detection — RunPod version.

Run:
    export HF_TOKEN=<your huggingface token>   # required, the dataset repo is private
    export DATASET_ROOT=/workspace/dataset     # optional, this is the default
    nohup python -u train_rtdetrv2_runpod.py > train.log 2>&1 &
    tail -f train.log

-u = unbuffered stdout, so `tail -f` actually shows progress live instead of
sitting empty until a buffer flushes. nohup + `&` = survives your SSH
session dropping; without it, closing the terminal kills a multi-hour run.

Everything through the COCO-JSON export mirrors dfine_s_640_kaggle.ipynb /
rtdetrv2_r18vd_640_kaggle.ipynb exactly, just with Kaggle-specific paths
swapped for RunPod ones. See that notebook's intro cell for the full
rationale behind the framework choice, the single-GPU pin, and AMP being
off.

Dataset: auto-downloaded from the private HF dataset repo (HF_DATASET_REPO
below) into DATASET_ROOT (default /workspace/dataset) if not already there,
using HF_TOKEN — never hardcode the token itself in this file. Skips the
download if the expected structure is already present, so re-running this
script after a partial/failed run doesn't re-pull ~2.6GB every time.
Expected structure, same as every other notebook in this repo:
    <DATASET_ROOT>/<DAGM|GC10-DET|KolektorSDD2|MPDD|MTD|Severstal|VisA>/
        <small|medium|large>/{images,labels_yolo}/
"""

import os

# Force single-GPU. Must be set before `import torch`. Even on a multi-GPU
# pod, HF Trainer auto-wraps the model in torch.nn.DataParallel when more
# than one GPU is visible (this script is a single process, no
# accelerate launch/torchrun) — and DataParallel cannot correctly scatter
# this script's list-based `labels` across devices, crashing inside
# RTDetrV2's denoising-query concat with a batch-size mismatch.
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
# Config
# ============================================================
RUN_NAME = "rtdetrv2_r18vd_imgsz640"
MODEL_LABEL = "RT-DETRv2-R18VD"
CHECKPOINT = "PekingU/rtdetr_v2_r18vd"
HF_DATASET_REPO = "Smalldefect/SmallDefectDataseet"  # private — needs HF_TOKEN

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 100
PATIENCE = 15
WORKERS = 2
CONF_THRESH = 0.25
MATCH_IOU = 0.5

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/rtdetrv2_run"))

# Redirect HF's model cache onto the /workspace volume. Left at its default
# (~/.cache/huggingface), it lands on the container disk instead — which is
# usually the smaller, less-expandable of the two on a RunPod pod, and
# mostly already consumed by the base CUDA/PyTorch image.
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))

PREPARED_DATASET_DIR = BASE_DIR / "yolo_dataset"
COCO_ROOT = BASE_DIR / "coco"
RUN_DIR = BASE_DIR / "runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test", "test_small", "test_medium", "test_large")

random.seed(SEED)

# ============================================================
# Dataset: download from the private HF repo if not already present,
# then validate. Nothing below this block should fail silently or hang
# waiting on input — this runs unattended over SSH.
# ============================================================
def missing_dataset_dirs():
    return [
        f"{name}/{size}"
        for name in DATASET_NAMES
        for size in SIZE_BUCKETS
        if not (DATASET_ROOT / name / size / "images").exists()
        or not (DATASET_ROOT / name / size / "labels_yolo").exists()
    ]


print(f"Checking dataset at {DATASET_ROOT} ...")
missing = missing_dataset_dirs()

if missing:
    print(f"Dataset incomplete/missing at {DATASET_ROOT} ({len(missing)} "
          f"bucket(s) missing) — attempting download from {HF_DATASET_REPO}.")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "HF_TOKEN is not set and the dataset isn't already on this pod. "
            "The dataset repo is private, so downloading it requires a token:\n"
            "  export HF_TOKEN=<your huggingface token>\n"
            "then re-run. (Get one at huggingface.co -> Settings -> Access Tokens.)"
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "--no-cache-dir", "huggingface_hub"],
        check=True,
    )
    # This repo is ~38k small files. Default concurrency (file-level
    # max_workers + Xet's own per-file range-get concurrency) is enough to
    # trip HF's rate limit partway through (already happened once — a 429
    # on the xet-read-token endpoint after ~2k files). Turn both down.
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
            print(
                f"Download attempt {attempt}/{DOWNLOAD_RETRIES} failed ({error}); "
                f"waiting {wait_seconds}s and retrying. snapshot_download resumes "
                "from already-fetched files, it doesn't restart from zero."
            )
            time.sleep(wait_seconds)
    print("Download finished. Re-checking structure...")
    missing = missing_dataset_dirs()

if missing:
    sys.exit(
        f"Dataset still incomplete at {DATASET_ROOT} after download. Missing "
        f"images/labels_yolo under: {', '.join(missing[:5])}"
        f"{' ...' if len(missing) > 5 else ''}\n"
        f"The repo's folder layout under {HF_DATASET_REPO} doesn't match what "
        "this script expects — check the actual structure on huggingface.co "
        "(each dataset folder should contain small/medium/large, each of "
        "those containing images/ and labels_yolo/) and adjust, or fix the "
        "layout at the source."
    )
print("Dataset structure looks complete.")

# ============================================================
# Install
# ============================================================
subprocess.run(
    # pandas/pillow added after confirming the hard way (D-FINE RunPod
    # script, same pod family) that they're not guaranteed preinstalled.
    [sys.executable, "-m", "pip", "install", "-q", "-U", "--no-cache-dir",
     "transformers", "accelerate", "torchmetrics", "pycocotools", "pyyaml",
     "pandas", "pillow"],
    check=True,
)
print("Installed.")

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ============================================================
# CUDA / numpy sanity check. A pip upgrade can pull in a numpy/torch combo
# that breaks CUDA silently (this exact class of bug already happened once
# on this project's Kaggle runs) — catch it here, in seconds, not partway
# through a multi-hour run.
# ============================================================
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
    sys.exit(f"Expected exactly 1 visible GPU, found {torch.cuda.device_count()} despite CUDA_VISIBLE_DEVICES=0 — investigate before continuing.")
DEVICE = "cuda"

# ============================================================
# Match images + labels (identical logic to the Kaggle notebooks)
# ============================================================
samples = []
missing_count = 0
for dataset_name in DATASET_NAMES:
    for size_bucket in SIZE_BUCKETS:
        image_dir = DATASET_ROOT / dataset_name / size_bucket / "images"
        label_dir = DATASET_ROOT / dataset_name / size_bucket / "labels_yolo"
        label_index = {p.stem: p for p in label_dir.glob("*.txt")}
        matched = 0
        bucket_missing = 0
        for image_path in image_dir.iterdir():
            # Skip macOS AppleDouble metadata sidecars (._<name>.<ext>) — they
            # share a real image's extension, so the suffix check alone lets
            # them through, doubling the dataset with unreadable junk files.
            if image_path.suffix.lower() not in IMAGE_EXTS or image_path.name.startswith("._"):
                continue
            image_stem = image_path.stem
            base_stem = image_stem.removesuffix("_defect")
            possible_label_stems = [image_stem, image_stem.replace("_defect", "_bbs"), f"{base_stem}_bbs"]
            label_path = next((label_index[s] for s in possible_label_stems if s in label_index), None)
            if label_path is None:
                bucket_missing += 1
                continue
            samples.append({
                "image_path": image_path, "label_path": label_path,
                "dataset": dataset_name, "size": size_bucket,
                "stratum": f"{dataset_name}_{size_bucket}",
            })
            matched += 1
        missing_count += bucket_missing
        print(f"{dataset_name}/{size_bucket}: {matched} matched, {bucket_missing} missing")

print("Total usable samples:", len(samples), "| missing:", missing_count)
if not samples:
    sys.exit("No image-label pairs matched — dataset folders exist but are empty or misnamed?")

# ============================================================
# Stratified split (identical seed/ratios to every other notebook here)
# ============================================================
by_stratum = defaultdict(list)
for sample in samples:
    by_stratum[sample["stratum"]].append(sample)

train_samples, val_samples, test_samples = [], [], []
rng = random.Random(SEED)
for stratum, group in sorted(by_stratum.items()):
    group = list(group)
    rng.shuffle(group)
    n = len(group)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_samples.extend(group[:n_train])
    val_samples.extend(group[n_train:n_train + n_val])
    test_samples.extend(group[n_train + n_val:])

rng.shuffle(train_samples)
rng.shuffle(val_samples)
rng.shuffle(test_samples)


def count_by_size(split_samples, split_name):
    counts = Counter(s["size"] for s in split_samples)
    print(f"{split_name}: total={len(split_samples)} small={counts['small']} medium={counts['medium']} large={counts['large']}")


count_by_size(train_samples, "Train")
count_by_size(val_samples, "Val")
count_by_size(test_samples, "Test")
assert train_samples and val_samples and test_samples

# ============================================================
# Export to YOLO-format dirs, then convert to COCO JSON
# ============================================================
def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


reset_dir(PREPARED_DATASET_DIR)
for split in SPLITS:
    (PREPARED_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (PREPARED_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def rewrite_label_as_single_class(src_label_path, dst_label_path):
    new_lines = []
    with open(src_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            new_lines.append("0 " + " ".join(parts[1:5]))
    with open(dst_label_path, "w") as f:
        f.write("\n".join(new_lines))


def export_split(split_name, split_samples):
    for idx, sample in enumerate(split_samples):
        src_image = sample["image_path"]
        src_label = sample["label_path"]
        safe_name = f"{sample['dataset']}_{sample['size']}_{idx}_{src_image.name}"
        dst_image = PREPARED_DATASET_DIR / "images" / split_name / safe_name
        dst_label = PREPARED_DATASET_DIR / "labels" / split_name / f"{Path(safe_name).stem}.txt"
        shutil.copy2(src_image, dst_image)
        rewrite_label_as_single_class(src_label, dst_label)


export_split("train", train_samples)
export_split("val", val_samples)
export_split("test", test_samples)
test_small_samples = [s for s in test_samples if s["size"] == "small"]
test_medium_samples = [s for s in test_samples if s["size"] == "medium"]
test_large_samples = [s for s in test_samples if s["size"] == "large"]
export_split("test_small", test_small_samples)
export_split("test_medium", test_medium_samples)
export_split("test_large", test_large_samples)
print("YOLO dataset prepared. Test overall:", len(test_samples),
      "| small:", len(test_small_samples), "| medium:", len(test_medium_samples),
      "| large:", len(test_large_samples))


def yolo_box_to_coco(parts, image_width, image_height):
    if len(parts) < 5:
        return None
    _, cx, cy, bw, bh = map(float, parts[:5])
    x1 = max(0.0, min((cx - bw / 2) * image_width, image_width))
    y1 = max(0.0, min((cy - bh / 2) * image_height, image_height))
    x2 = max(0.0, min((cx + bw / 2) * image_width, image_width))
    y2 = max(0.0, min((cy + bh / 2) * image_height, image_height))
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    return [round(x1, 4), round(y1, 4), round(width, 4), round(height, 4)]


def convert_split_to_coco(split_name):
    image_dir = PREPARED_DATASET_DIR / "images" / split_name
    label_dir = PREPARED_DATASET_DIR / "labels" / split_name
    images, annotations = [], []
    annotation_id = 1
    skipped_boxes = 0
    for image_id, image_path in enumerate(sorted(image_dir.iterdir()), start=1):
        if image_path.suffix.lower() not in IMAGE_EXTS or image_path.name.startswith("._"):
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = label_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text().splitlines():
            box = yolo_box_to_coco(line.split(), width, height)
            if box is None:
                skipped_boxes += 1
                continue
            annotations.append({
                "id": annotation_id, "image_id": image_id, "category_id": 0,
                "bbox": box, "area": round(box[2] * box[3], 4), "iscrowd": 0,
            })
            annotation_id += 1
    coco = {
        "info": {"description": "One-class small-defect RT-DETRv2 dataset"},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": 0, "name": "defect", "supercategory": "defect"}],
    }
    annotation_dir = COCO_ROOT / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    output_path = annotation_dir / f"instances_{split_name}.json"
    output_path.write_text(json.dumps(coco, indent=2))
    return {"split": split_name, "images": len(images), "instances": len(annotations), "skipped_boxes": skipped_boxes}


conversion_rows = [convert_split_to_coco(split) for split in SPLITS]
# Same deterministic split logic/seed as the Kaggle notebooks, so these
# counts should match them exactly. A mismatch means the transferred
# dataset differs, not necessarily a bug — warn rather than hard-fail.
expected_images = {"train": 8858, "val": 1892, "test": 1920, "test_small": 462, "test_medium": 868, "test_large": 590}
for row in conversion_rows:
    print(row)
    if row["images"] != expected_images[row["split"]]:
        print(f"NOTE: {row['split']} has {row['images']} images, expected {expected_images[row['split']]}.")
    assert row["instances"] > 0 and row["skipped_boxes"] == 0, row

# ============================================================
# Load RT-DETRv2
# ============================================================
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

id2label = {0: "defect"}
label2id = {"defect": 0}
image_processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT, size={"height": IMG_SIZE, "width": IMG_SIZE})
model = RTDetrV2ForObjectDetection.from_pretrained(
    CHECKPOINT, id2label=id2label, label2id=label2id, num_labels=len(id2label), ignore_mismatched_sizes=True,
)
model.to(DEVICE)
print("Loaded", CHECKPOINT, "on", DEVICE)


class CocoDefectDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, ann_file, image_processor):
        self.image_dir = Path(image_dir)
        self.image_processor = image_processor
        coco = json.loads(Path(ann_file).read_text())
        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())
        self.annotations_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations_by_image[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image = Image.open(self.image_dir / image_info["file_name"]).convert("RGB")
        annotations = {"image_id": image_id, "annotations": self.annotations_by_image[image_id]}
        encoded = self.image_processor(images=image, annotations=annotations, return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"][0], "labels": encoded["labels"][0]}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }


train_dataset = CocoDefectDataset(
    PREPARED_DATASET_DIR / "images" / "train", COCO_ROOT / "annotations" / "instances_train.json", image_processor,
)
val_dataset = CocoDefectDataset(
    PREPARED_DATASET_DIR / "images" / "val", COCO_ROOT / "annotations" / "instances_val.json", image_processor,
)
print("Train images:", len(train_dataset), "| Val images:", len(val_dataset))

# ============================================================
# Fixed-threshold matching helpers (Precision/Recall/FP at conf=0.25,
# IoU=0.5 — same policy as the rest of this project's runs)
# ============================================================
def box_iou_xyxy(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return torch.zeros((len(box1), len(box2)))
    x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-6)


def match_predictions(pred_boxes, pred_scores, gt_boxes, conf_thresh=CONF_THRESH, iou_thresh=MATCH_IOU):
    keep = pred_scores >= conf_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    ious = box_iou_xyxy(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    for pred_idx in torch.argsort(-pred_scores):
        best_gt = int(torch.argmax(ious[pred_idx]).item())
        best_iou = float(ious[pred_idx, best_gt].item())
        if best_iou >= iou_thresh and best_gt not in matched_gt:
            matched_gt.add(best_gt)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ============================================================
# Train
# ============================================================
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=str(RUN_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    dataloader_num_workers=WORKERS,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_steps=300,
    lr_scheduler_type="cosine",
    fp16=False,  # D-FINE (same DETR family) hit a NaN-box failure under AMP on this dataset
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    seed=SEED,
    report_to="none",
    run_name=RUN_NAME,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

train_result = trainer.train()
print(train_result)
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best eval_loss:", trainer.state.best_metric)
print("Stopped after epoch:", trainer.state.epoch, "/ budget", EPOCHS)

# ============================================================
# Evaluation: val, overall test, and small/medium/large test subsets
# ============================================================
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@torch.no_grad()
def run_split_evaluation(image_dir, ann_file, low_thresh=0.001):
    image_dir = Path(image_dir)
    coco = json.loads(Path(ann_file).read_text())
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    model.eval()
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    total_tp = total_fp = total_fn = 0
    for image_id, image_info in images.items():
        image = Image.open(image_dir / image_info["file_name"]).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        result = image_processor.post_process_object_detection(
            outputs, threshold=low_thresh, target_sizes=torch.tensor([(image.height, image.width)]),
        )[0]
        pred_boxes = result["boxes"].cpu()
        pred_scores = result["scores"].cpu()
        gt = [ann["bbox"] for ann in anns_by_image[image_id]]
        gt_boxes_xywh = torch.tensor(gt, dtype=torch.float32) if gt else torch.zeros((0, 4))
        gt_boxes_xyxy = gt_boxes_xywh.clone()
        if len(gt_boxes_xyxy):
            gt_boxes_xyxy[:, 2] = gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2]
            gt_boxes_xyxy[:, 3] = gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3]
        map_metric.update(
            [{"boxes": pred_boxes, "scores": pred_scores, "labels": torch.zeros(len(pred_boxes), dtype=torch.int)}],
            [{"boxes": gt_boxes_xyxy, "labels": torch.zeros(len(gt_boxes_xyxy), dtype=torch.int)}],
        )
        tp, fp, fn = match_predictions(pred_boxes, pred_scores, gt_boxes_xyxy)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    map_result = map_metric.compute()
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    return {
        "mAP50": float(map_result["map_50"]), "mAP50_95": float(map_result["map"]),
        "Precision": precision, "Recall": recall,
        "FP_Per_Image": total_fp / max(len(images), 1), "num_images": len(images),
    }


eval_sets = {
    "val": (PREPARED_DATASET_DIR / "images" / "val", COCO_ROOT / "annotations" / "instances_val.json"),
    "overall": (PREPARED_DATASET_DIR / "images" / "test", COCO_ROOT / "annotations" / "instances_test.json"),
    "small": (PREPARED_DATASET_DIR / "images" / "test_small", COCO_ROOT / "annotations" / "instances_test_small.json"),
    "medium": (PREPARED_DATASET_DIR / "images" / "test_medium", COCO_ROOT / "annotations" / "instances_test_medium.json"),
    "large": (PREPARED_DATASET_DIR / "images" / "test_large", COCO_ROOT / "annotations" / "instances_test_large.json"),
}
results = {}
for name, (image_dir, ann_file) in eval_sets.items():
    print("Evaluating:", name)
    results[name] = run_split_evaluation(image_dir, ann_file)
    print(results[name])

# ============================================================
# Inference-time benchmark (forward pass only, excludes preprocessing —
# matches how the YOLO-based rows in the project's comparison table report
# this column)
# ============================================================
def measure_inference_time_ms(image_paths, n_warmup=5, n_measure=50):
    model.eval()
    reps = (n_warmup + n_measure) // len(image_paths) + 1
    sample_paths = (image_paths * reps)[: n_warmup + n_measure]
    prepared = []
    for path in sample_paths:
        image = Image.open(path).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        prepared.append(inputs)
    with torch.no_grad():
        for inputs in prepared[:n_warmup]:
            model(**inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for inputs in prepared[n_warmup:]:
            model(**inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / n_measure) * 1000


test_image_paths = sorted((PREPARED_DATASET_DIR / "images" / "test").iterdir())
inference_time_ms = measure_inference_time_ms(test_image_paths)
print(f"Inference time: {inference_time_ms:.2f} ms/image (model forward pass only, excludes preprocessing)")

# ============================================================
# Final summary export
# ============================================================
overall, small, medium, large = (results[k] for k in ("overall", "small", "medium", "large"))
summary_row = {
    "Experiment": RUN_NAME,
    "Model": MODEL_LABEL,
    "Batch": BATCH_SIZE,
    "Epochs": round(trainer.state.epoch),
    "mAP50": overall["mAP50"],
    "mAP50_95": overall["mAP50_95"],
    "Precision": overall["Precision"],
    "Recall": overall["Recall"],
    "mAP50_Small": small["mAP50"],
    "mAP50_Medium": medium["mAP50"],
    "mAP50_Large": large["mAP50"],
    "Recall_Small": small["Recall"],
    "Recall_Medium": medium["Recall"],
    "Recall_Large": large["Recall"],
    "Inference_Time_ms": inference_time_ms,
    "FP_Per_Image": overall["FP_Per_Image"],
    "Notes": (
        f"RT-DETRv2-R18VD via HF transformers on RunPod, early stopping on eval_loss "
        f"(patience={PATIENCE}, epoch budget {EPOCHS}), fixed 640 split, P/R/FP at conf={CONF_THRESH}"
    ),
}

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)
pd.DataFrame([{"split": k, **v} for k, v in results.items()]).to_csv(
    FINAL_OUTPUT_DIR / "evaluation_metrics.csv", index=False
)
pd.DataFrame(conversion_rows).to_csv(FINAL_OUTPUT_DIR / "split_counts.csv", index=False)
trainer.save_model(str(FINAL_OUTPUT_DIR / "best_checkpoint"))
image_processor.save_pretrained(str(FINAL_OUTPUT_DIR / "best_checkpoint"))
shutil.copytree(COCO_ROOT / "annotations", FINAL_OUTPUT_DIR / "coco_annotations", dirs_exist_ok=True)

print("Saved final artifacts to:", FINAL_OUTPUT_DIR)
print(summary_df.to_string(index=False))
print("DONE.")
