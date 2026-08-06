"""
D-FINE-S, 640 px, one-class small-defect detection — RunPod version.

Faithful port of dfine_s_640_kaggle.ipynb: same dataset split, same COCO
conversion, same batch-scaled LR, same "no AMP" fix for the NaN-box failure
that notebook already hit once, same official repo (Peterande/D-FINE) and
its own train.py — not a reimplementation, not HF transformers. Only the
paths changed (Kaggle -> RunPod) and the dataset now auto-downloads from
Hugging Face instead of being an attached Kaggle Dataset.

Run:
    export HF_TOKEN=<your huggingface token>   # required, the dataset repo is private
    export DATASET_ROOT=/workspace/dataset     # optional, this is the default
    nohup python -u train_dfine_s_runpod.py > train.log 2>&1 &
    tail -f train.log

Same params as the Kaggle notebook, unchanged:
    EPOCHS=50, BATCH_SIZE=8, no early stopping (the Kaggle notebook never had
    any), LR 5e-5 / backbone LR 2.5e-5 (the official D-FINE-S config's own
    4e-4 @ global-batch-64, linearly scaled to batch 8), AMP off.
"""

import os

# Defensive, same reasoning as the RT-DETRv2 RunPod script even though
# D-FINE's own train.py (run here as plain `python train.py`, not via
# torchrun) shouldn't auto-engage multi-GPU on its own — costs nothing if
# it wasn't going to happen anyway, and today already had one surprise
# multi-GPU auto-wrap (HF Trainer + DataParallel) that a pin like this
# would have prevented outright.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ast
import json
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# Config — same values as dfine_s_640_kaggle.ipynb
# ============================================================
RUN_NAME = "RunJ_dfine_s_imgsz640"
MODEL_LABEL = "D-FINE-S"
HF_DATASET_REPO = "Smalldefect/SmallDefectDataseet"  # private — needs HF_TOKEN

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 50
WORKERS = 2

# D-FINE-S's official custom config is tuned for global batch 64 and lr 4e-4.
# This is the linear batch-8 equivalent: 4e-4 * 8 / 64 = 5e-5. Unchanged from
# the Kaggle notebook.
BASE_LR = 0.00005
BACKBONE_LR = 0.000025

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/dfine_run"))

os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))

PREPARED_DATASET_DIR = BASE_DIR / "yolo_dataset"
DFINE_REPO = BASE_DIR / "D-FINE"
COCO_ROOT = BASE_DIR / "coco"
RUN_DIR = BASE_DIR / "dfine_runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test", "test_small", "test_medium", "test_large")

random.seed(SEED)

# ============================================================
# numpy/pandas/Pillow are not guaranteed preinstalled on every RunPod base
# image (confirmed the hard way — pandas was missing on this exact pod).
# torch is not: if that's missing this template is unusable anyway, so it's
# not worth defending against.
# ============================================================
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "numpy", "pandas", "pillow"],
    check=True,
)

# ============================================================
# CUDA / numpy sanity check. Same class of bug already happened once on
# this project's Kaggle runs ("ms detr") — catch it here, in seconds, not
# partway through a run.
# ============================================================
import numpy as np
import pandas as pd
import torch
from PIL import Image

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

# ============================================================
# Dataset: download from the private HF repo if not already present,
# then validate. Same logic as the RT-DETRv2 RunPod script, including the
# reduced concurrency + retry-with-backoff — this repo is ~38k small files
# and tripped HF's rate limit once already at default concurrency.
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
            "then re-run. (Get one at huggingface.co -> Settings -> Access Tokens.)\n"
            "Alternatively, scp a pre-built dataset.tar to this pod and extract it "
            "to DATASET_ROOT yourself before running this script — it'll skip "
            "the download entirely if the expected structure is already there."
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
        f"Dataset still incomplete at {DATASET_ROOT}. Missing images/labels_yolo "
        f"under: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}\n"
        f"Check the actual structure under {HF_DATASET_REPO} on huggingface.co "
        "(each dataset folder should contain small/medium/large, each of those "
        "containing images/ and labels_yolo/) and adjust, or fix the layout at "
        "the source."
    )
print("Dataset structure looks complete.")

# ============================================================
# Match images + labels (identical logic to every other notebook/script
# in this repo)
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
# Export to YOLO-format dirs (single class), then convert to COCO JSON —
# same two-step pipeline as the Kaggle notebook, which itself reuses the
# exact split the YOLO baselines use for comparability.
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
        "info": {"description": "One-class small-defect D-FINE dataset"},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": 0, "name": "defect", "supercategory": "defect"}],
    }
    annotation_dir = COCO_ROOT / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    output_path = annotation_dir / f"instances_{split_name}.json"
    output_path.write_text(json.dumps(coco, indent=2))
    return {"split": split_name, "images": len(images), "instances": len(annotations), "skipped_boxes": skipped_boxes}


conversion_rows = [convert_split_to_coco(split) for split in SPLITS]
conversion_by_split = {row["split"]: row for row in conversion_rows}
expected_images = {"train": 8858, "val": 1892, "test": 1920, "test_small": 462, "test_medium": 868, "test_large": 590}
for row in conversion_rows:
    print(row)
    if row["images"] != expected_images[row["split"]]:
        print(f"NOTE: {row['split']} has {row['images']} images, expected {expected_images[row['split']]}.")
    assert row["instances"] > 0 and row["skipped_boxes"] == 0, row

# ============================================================
# Clone the official D-FINE repo (idempotent) and install its deps
# ============================================================
if not DFINE_REPO.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Peterande/D-FINE.git", str(DFINE_REPO)],
        check=True,
    )
subprocess.run(
    # matplotlib added explicitly: D-FINE's own src/solver/validator.py
    # imports it directly (confirmed by an actual crash on this exact run —
    # ModuleNotFoundError at that import), but it isn't in requirements.txt,
    # almost certainly because the repo's own examples assume Kaggle/Colab,
    # which preinstall it. Same class of gap as pandas earlier.
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
     "-r", str(DFINE_REPO / "requirements.txt"), "pycocotools", "matplotlib"],
    check=True,
)
print("D-FINE ready at:", DFINE_REPO)

# ============================================================
# Write the D-FINE config. Same stability fixes as the Kaggle notebook:
# batch-scaled LR and full precision (no AMP) — the earlier AMP run on
# this exact dataset produced NaN predicted boxes.
# ============================================================
CONFIG_DIR = DFINE_REPO / "configs" / "dfine" / "custom"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)  # write_text() doesn't create parent dirs
TRAIN_CONFIG = CONFIG_DIR / "dfine_defect_s_stable.yml"


def yaml_float(value):
    """Format a float so PyYAML's resolver always parses it back as a float,
    not a string. PyYAML's float regex requires a literal '.' in the
    mantissa even in scientific notation — Python's own str(5e-05) omits it
    ('5e-05'), which PyYAML then silently loads as a plain string instead
    of raising, and that string later blows up wherever the value is used
    numerically. This is what crashed AdamW's lr on this exact run.
    """
    text = repr(float(value))
    if "e" in text and "." not in text.split("e")[0]:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}.0e{exponent}"
    return text


def write_config(path, image_split, annotation_split, output_dir):
    image_dir = PREPARED_DATASET_DIR / "images" / image_split
    annotation_file = COCO_ROOT / "annotations" / f"instances_{annotation_split}.json"
    path.write_text(f'''__include__: [ './dfine_hgnetv2_s_custom.yml' ]

output_dir: {output_dir}
num_classes: 1
remap_mscoco_category: False
epochs: {EPOCHS}

# Do not enable AMP. The earlier AMP run produced NaN predicted boxes.
use_amp: False
scaler:
  enabled: False

optimizer:
  type: AdamW
  lr: {yaml_float(BASE_LR)}
  betas: [0.9, 0.999]
  weight_decay: 0.0001
  params:
    - params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: {yaml_float(BACKBONE_LR)}
    - params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: {yaml_float(BACKBONE_LR)}
      weight_decay: 0.0
    - params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0

train_dataloader:
  total_batch_size: {BATCH_SIZE}
  num_workers: {WORKERS}
  dataset:
    img_folder: {PREPARED_DATASET_DIR / 'images' / 'train'}
    ann_file: {COCO_ROOT / 'annotations' / 'instances_train.json'}

val_dataloader:
  total_batch_size: {BATCH_SIZE}
  num_workers: {WORKERS}
  dataset:
    img_folder: {image_dir}
    ann_file: {annotation_file}
''')


write_config(TRAIN_CONFIG, "val", "val", RUN_DIR)
print(TRAIN_CONFIG.read_text())

# ============================================================
# Train — full-precision D-FINE-S. Do not add --use-amp.
# ============================================================
command = [sys.executable, "train.py", "-c", str(TRAIN_CONFIG), "--seed", str(SEED)]
print("Running:", " ".join(command))
subprocess.run(command, cwd=DFINE_REPO, check=True)

# ============================================================
# Locate the best checkpoint
# ============================================================
checkpoints = sorted(RUN_DIR.rglob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
if not checkpoints:
    raise FileNotFoundError(f"No D-FINE checkpoint found under {RUN_DIR}")
best_named = [item for item in checkpoints if "best" in item.name.lower()]
BEST_CHECKPOINT = best_named[0] if best_named else checkpoints[0]
print("Using checkpoint:", BEST_CHECKPOINT)

# ============================================================
# Evaluate the same checkpoint on val + overall/small/medium/large test
# ============================================================
evaluation_sets = {
    "val": ("val", "val"),
    "overall": ("test", "test"),
    "small": ("test_small", "test_small"),
    "medium": ("test_medium", "test_medium"),
    "large": ("test_large", "test_large"),
}

EVAL_LOG_DIR = FINAL_OUTPUT_DIR / "evaluation_logs"
EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
evaluation_logs = {}
for name, (image_split, annotation_split) in evaluation_sets.items():
    eval_config = CONFIG_DIR / f"dfine_defect_s_{name}.yml"
    eval_output_dir = BASE_DIR / "dfine_evaluations" / RUN_NAME / name
    write_config(eval_config, image_split, annotation_split, eval_output_dir)
    command = [sys.executable, "train.py", "-c", str(eval_config), "--test-only", "-r", str(BEST_CHECKPOINT)]
    print("Running evaluation:", name)
    completed = subprocess.run(command, cwd=DFINE_REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path = EVAL_LOG_DIR / f"{name}.log"
    log_path.write_text(completed.stdout)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} evaluation failed. Read {log_path}")
    evaluation_logs[name] = log_path

# ============================================================
# Parse D-FINE's own validator output + COCO metrics
# ============================================================
def parse_evaluation(log_path):
    text = Path(log_path).read_text()
    validator_match = re.findall(r"Metrics:\s*(\{.*?\})", text)
    validator = ast.literal_eval(validator_match[-1]) if validator_match else {}
    # No re.S here on purpose: each stat is one line, and DOTALL previously let
    # the match drift across lines (even into the Recall block) looking for
    # its next literal. And the capture anchors on "] =" specifically, not
    # bare "=" — bare "=" matches the FIRST equals sign on the line, which is
    # the one inside "maxDets=100", not the real value at the end. That's the
    # exact bug that produced mAP50=100.0 on this run's actual output: it
    # captured "100" out of "maxDets=100" instead of the real 0.0xx value.
    ap50_95 = re.findall(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s*all.*?\]\s*=\s*([0-9.]+)", text)
    ap50 = re.findall(r"Average Precision.*?IoU=0\.50\s*\|.*?area=\s*all.*?\]\s*=\s*([0-9.]+)", text)
    return {
        "mAP50": float(ap50[-1]) if ap50 else float("nan"),
        "mAP50_95": float(ap50_95[-1]) if ap50_95 else float("nan"),
        "Precision": validator.get("precision", float("nan")),
        "Recall": validator.get("recall", float("nan")),
        "FPs": validator.get("FPs", float("nan")),
    }


metrics = {name: parse_evaluation(path) for name, path in evaluation_logs.items()}
overall, small, medium, large = (metrics[key] for key in ("overall", "small", "medium", "large"))

test_image_count = conversion_by_split["test"]["images"]
summary_row = {
    "Experiment": RUN_NAME,
    "Model": MODEL_LABEL,
    "Batch": BATCH_SIZE,
    "Epochs": EPOCHS,
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
    "Inference_Time_ms": float("nan"),
    "FP_per_Image": overall["FPs"] / test_image_count if pd.notna(overall["FPs"]) else float("nan"),
    "Notes": "D-FINE-S via official Peterande/D-FINE repo on RunPod, full precision, batch-scaled LR, fixed 640 split",
}

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)
pd.DataFrame([{"split": s, **v} for s, v in metrics.items()]).to_csv(FINAL_OUTPUT_DIR / "evaluation_metrics.csv", index=False)
pd.DataFrame(conversion_rows).to_csv(FINAL_OUTPUT_DIR / "split_counts.csv", index=False)
shutil.copy2(TRAIN_CONFIG, FINAL_OUTPUT_DIR / TRAIN_CONFIG.name)
shutil.copy2(BEST_CHECKPOINT, FINAL_OUTPUT_DIR / "best_checkpoint.pth")
shutil.copytree(COCO_ROOT / "annotations", FINAL_OUTPUT_DIR / "coco_annotations", dirs_exist_ok=True)

print("Saved final artifacts to:", FINAL_OUTPUT_DIR)
print(summary_df.to_string(index=False))
print("DONE.")
