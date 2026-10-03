"""Train and evaluate YOLO-SM on the project’s combined one-class dataset.

The script works with the already-prepared dataset used by the existing YOLO
baselines and can also prepare that layout from the raw seven-dataset layout.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from yolosm_losses import yolosm_loss
from yolosm_metrics import DetectionAccumulator, nms, summarize_accumulator
from yolosm_model import YOLOSM, count_parameters


DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED = int(os.environ.get("SEED", "42"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
EPOCHS = int(os.environ.get("EPOCHS", "3" if os.environ.get("SMOKE") == "1" else "300"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
PATIENCE = int(os.environ.get("PATIENCE", "40"))
MOSAIC_PROB = float(os.environ.get("MOSAIC_PROB", "0.5" if os.environ.get("SMOKE") != "1" else "0"))
LR_MAX = float(os.environ.get("LR_MAX", "0.001"))
LR_MIN = float(os.environ.get("LR_MIN", "0.000001"))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "5"))
DEVICE = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
RUN_NAME = os.environ.get("RUN_NAME", f"RunY_yolosm_imgsz{IMG_SIZE}")
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/yolosm"))
RUN_DIR = BASE_DIR / "runs" / RUN_NAME
YOLO_DATASET_ROOT = Path(os.environ["YOLO_DATASET_ROOT"]) if os.environ.get("YOLO_DATASET_ROOT") else None
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def candidate_stems(stem: str) -> list[str]:
    base = stem.removesuffix("_defect")
    return [stem, stem.replace("_defect", "_bbs"), base, f"{base}_bbs"]


def is_prepared(root: Path) -> bool:
    return all((root / "images" / split).exists() and (root / "labels" / split).exists() for split in ("train", "val", "test"))


def write_single_class_label(source: Path, destination: Path) -> None:
    lines = []
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                lines.append("0 " + " ".join(parts[1:5]))
    destination.write_text("\n".join(lines), encoding="utf-8")


def prepare_from_source(source_root: Path, output_root: Path) -> Path:
    """Create a fixed-stratified one-class YOLO dataset from raw project folders."""
    samples = []
    for dataset in DATASET_NAMES:
        for size in SIZE_BUCKETS:
            image_dir = source_root / dataset / size / "images"
            label_dir = source_root / dataset / size / "labels_yolo"
            if not image_dir.exists() or not label_dir.exists():
                continue
            labels = {p.stem: p for p in label_dir.glob("*.txt")}
            for image in sorted(image_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_EXTS or image.name.startswith("._"):
                    continue
                label = next((labels[s] for s in candidate_stems(image.stem) if s in labels), None)
                if label is not None:
                    samples.append({"image": image, "label": label, "dataset": dataset, "size": size})
    if not samples:
        raise FileNotFoundError(f"No source samples found under {source_root}")

    by_stratum = defaultdict(list)
    for sample in samples:
        by_stratum[f"{sample['dataset']}_{sample['size']}"].append(sample)
    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for group in by_stratum.values():
        group = list(group)
        rng.shuffle(group)
        n_train = int(len(group) * 0.70)
        n_val = int(len(group) * 0.15)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, split_samples in splits.items():
        image_out = output_root / "images" / split
        label_out = output_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(split_samples):
            name = f"{sample['dataset']}_{sample['size']}_{idx}_{sample['image'].name}"
            dst_image = image_out / name
            dst_label = label_out / f"{Path(name).stem}.txt"
            shutil.copy2(sample["image"], dst_image)
            write_single_class_label(sample["label"], dst_label)
            manifest.append({"split": split, "file": name, "dataset": sample["dataset"], "size": sample["size"]})
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_root


def resolve_dataset_root() -> Path:
    if YOLO_DATASET_ROOT is not None and is_prepared(YOLO_DATASET_ROOT):
        return YOLO_DATASET_ROOT
    for candidate in [DATASET_ROOT / "run_a_yolo_dataset", DATASET_ROOT, Path("/kaggle/input/run_a_yolo_dataset")]:
        if is_prepared(candidate):
            return candidate
    prepared = BASE_DIR / "prepared_dataset"
    return prepare_from_source(DATASET_ROOT, prepared)


def infer_metadata(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    for dataset in sorted(DATASET_NAMES, key=len, reverse=True):
        prefix = f"{dataset}_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            size = remainder.split("_", 1)[0]
            if size in SIZE_BUCKETS:
                return dataset, size
    return "unknown", "unknown"


def load_boxes(label_path: Path) -> Tensor:
    rows = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[1:5]])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 4), dtype=torch.float32)


class DefectDataset(Dataset):
    def __init__(self, root: Path, split: str, training: bool = False) -> None:
        self.root = root
        self.split = split
        self.training = training
        self.mosaic_prob = MOSAIC_PROB if training else 0.0
        self.images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._"))
        self.labels = root / "labels" / split
        manifest_path = root / "manifest.json"
        manifest = {row["file"]: row for row in json.loads(manifest_path.read_text(encoding="utf-8"))} if manifest_path.exists() else {}
        self.meta = {p.name: (manifest.get(p.name, {}).get("dataset"), manifest.get(p.name, {}).get("size")) for p in self.images}
        for p in self.images:
            if not self.meta[p.name][0] or not self.meta[p.name][1]:
                self.meta[p.name] = infer_metadata(p.name)

    def __len__(self) -> int:
        return len(self.images)

    def _load(self, index: int) -> tuple[Tensor, Tensor, str, str]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        boxes = load_boxes(self.labels / f"{image_path.stem}.txt")
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        dataset, size = self.meta[image_path.name]
        return pixels, boxes, dataset or "unknown", size or "unknown"

    def _mosaic(self, index: int) -> tuple[Tensor, Tensor, str, str]:
        tile_size = IMG_SIZE // 2
        canvas = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        all_boxes = []
        for tile_idx, item_index in enumerate([index] + [random.randrange(len(self.images)) for _ in range(3)]):
            pixels, boxes, dataset, size = self._load(item_index)
            pixels = torch.nn.functional.interpolate(pixels.unsqueeze(0), size=(tile_size, tile_size), mode="bilinear", align_corners=False)[0]
            row, col = divmod(tile_idx, 2)
            y0, x0 = row * tile_size, col * tile_size
            canvas[:, y0:y0 + tile_size, x0:x0 + tile_size] = pixels
            if len(boxes):
                b = boxes.clone()
                b[:, 0] = (b[:, 0] * tile_size + x0) / IMG_SIZE
                b[:, 1] = (b[:, 1] * tile_size + y0) / IMG_SIZE
                b[:, 2] = b[:, 2] * tile_size / IMG_SIZE
                b[:, 3] = b[:, 3] * tile_size / IMG_SIZE
                all_boxes.append(b)
        return canvas, torch.cat(all_boxes, dim=0) if all_boxes else torch.zeros((0, 4)), "mosaic", "mixed"

    def __getitem__(self, index: int) -> dict:
        if self.training and random.random() < self.mosaic_prob:
            pixels, boxes, dataset, size = self._mosaic(index)
        else:
            pixels, boxes, dataset, size = self._load(index)
        if self.training and random.random() < 0.5:
            pixels = torch.flip(pixels, dims=[2])
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]
        return {"image": pixels, "boxes": boxes, "dataset": dataset, "size": size}


def collate(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([row["image"] for row in batch]),
        "boxes": [row["boxes"] for row in batch],
        "datasets": [row["dataset"] for row in batch],
        "sizes": [row["size"] for row in batch],
    }


def set_learning_rate(optimizer: Adam, step: int, total_steps: int, warmup_steps: int) -> float:
    """Apply the paper-style linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        fraction = step / max(warmup_steps, 1)
        lr = LR_MAX * (0.1 + 0.9 * fraction)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def predict_records(model: YOLOSM, loader: DataLoader, device: torch.device) -> DetectionAccumulator:
    accumulator = DetectionAccumulator()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["images"].to(device, non_blocking=True))
            scores = outputs["obj_logits"].sigmoid() * outputs["cls_logits"].sigmoid()
            for idx in range(scores.shape[0]):
                score = scores[idx, :, 0]
                keep = score >= 0.001
                boxes = outputs["boxes"][idx][keep].clamp(0, IMG_SIZE)
                score = score[keep]
                if len(score):
                    keep_idx = nms(boxes, score, iou_threshold=0.3)
                    boxes, score = boxes[keep_idx], score[keep_idx]
                gt = batch["boxes"][idx].to(device)
                if len(gt):
                    gt = gt * IMG_SIZE
                    gt = torch.stack([gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2, gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2], dim=1)
                accumulator.add(boxes, score, gt, batch["datasets"][idx], batch["sizes"][idx])
    return accumulator


def main() -> None:
    seed_everything(SEED)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    root = resolve_dataset_root()
    print(f"Dataset: {root}")
    train_loader = DataLoader(DefectDataset(root, "train", True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0, collate_fn=collate)
    val_loader = DataLoader(DefectDataset(root, "val"), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0, collate_fn=collate)
    test_loader = DataLoader(DefectDataset(root, "test"), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0, collate_fn=collate)

    model = YOLOSM(num_classes=1).to(DEVICE)
    print(f"Device: {DEVICE} | trainable parameters: {count_parameters(model):,}")
    optimizer = Adam(model.parameters(), lr=LR_MAX, betas=(0.9, 0.999))
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(EPOCHS * steps_per_epoch, 1)
    warmup_steps = min(WARMUP_EPOCHS * steps_per_epoch, total_steps - 1) if total_steps > 1 else 0
    history = []
    best_map = -1.0
    stale = 0
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config = {"run": RUN_NAME, "dataset_root": str(root), "img_size": IMG_SIZE, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED, "mosaic_prob": MOSAIC_PROB, "lr_max": LR_MAX, "lr_min": LR_MIN, "warmup_epochs": WARMUP_EPOCHS, "parameters": count_parameters(model), "device": str(DEVICE)}
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    try:
        import wandb
        wandb_mode = os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "disabled")
        wandb.init(project="smallDefectDetection", name=RUN_NAME, config=config, mode=wandb_mode)
    except Exception as exc:
        wandb = None
        print(f"W&B disabled: {exc}")

    global_step = 0
    lr = LR_MAX
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        running = defaultdict(float)
        for batch in train_loader:
            lr = set_learning_rate(optimizer, global_step, total_steps, warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["images"].to(DEVICE, non_blocking=True))
            loss, parts = yolosm_loss(outputs, [boxes.to(DEVICE) for boxes in batch["boxes"]])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            global_step += 1
            for key, value in parts.items():
                running[key] += value
        val_acc = predict_records(model, val_loader, DEVICE)
        val = val_acc.compute()
        row = {"epoch": epoch, "train_loss": running["loss"] / max(len(train_loader), 1), "val_mAP50": val["mAP50"], "val_mAP50_95": val["mAP50_95"], "val_precision": val["precision"], "val_recall": val["recall"], "lr": lr, "seconds": time.perf_counter() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(RUN_DIR / "training_history.csv", index=False)
        print(f"Epoch {epoch:03d}/{EPOCHS} | loss={row['train_loss']:.4f} | val mAP50={row['val_mAP50']:.4f} | val recall={row['val_recall']:.4f} | {row['seconds']:.1f}s")
        if wandb is not None:
            wandb.log(row, step=epoch)
        val_map = float(row["val_mAP50"]) if np.isfinite(row["val_mAP50"]) else 0.0
        if val_map > best_map:
            best_map = val_map
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "config": config, "val_metrics": val}, RUN_DIR / "best_model.pt")
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping after {PATIENCE} stale epochs.")
            break

    checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc = predict_records(model, test_loader, DEVICE)
    rows = summarize_accumulator(test_acc)
    pd.DataFrame(rows).to_csv(RUN_DIR / "evaluation_metrics.csv", index=False)
    overall = next(row for row in rows if row["split"] == "overall")
    summary = {"Experiment": RUN_NAME, "Model": "YOLO-SM", "Epochs": checkpoint["epoch"], "mAP50": overall["mAP50"], "mAP50_95": overall["mAP50_95"], "Precision": overall["precision"], "Recall": overall["recall"], "Inference_Time_ms": None, "Notes": "Custom PyTorch YOLO-SM reproduction; one-class full combined dataset."}
    pd.DataFrame([summary]).to_csv(RUN_DIR / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if wandb is not None:
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
