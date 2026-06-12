"""
DAGM preprocessing: weak ellipse mask PNGs → tight bounding boxes via contours → YOLO + COCO.

Kaggle input:  /kaggle/input/dagm-2007-competition-dataset-opticalinspection/
               Class1/ through Class10/, each with Train/ and Test/.
               Defective images have matching masks in Train/Label/ or Test/Label/.
               Mask file has same name as the defective image (e.g. 0001.PNG → Label/0001.PNG).

Strategy: threshold the ellipse mask PNG → findContours → boundingRect → tight BB.
We process Train AND Test defective images (no split preserved).

Output layout:
  processed/DAGM/
    images/       dagm_{class}_{nnnn}_defect.png
    masks/        dagm_{class}_{nnnn}_mask.png
    labels_yolo/  dagm_{class}_{nnnn}_bbs.txt
    labels_coco/  {class_name}.json
    class_mapping.txt
    size_manifest.csv
"""

import shutil
from pathlib import Path

import cv2
from PIL import Image

from utils import (
    COCOBuilder, append_manifest, bbox_to_yolo,
    categorize_defect, init_manifest,
)

DAGM_INPUT = Path("/kaggle/input/dagm-2007-competition-dataset-opticalinspection")
OUTPUT_ROOT = Path("/kaggle/working/processed/DAGM")

# DAGM has 10 texture classes
NUM_CLASSES = 10


def _ellipse_mask_to_bbox(mask_path: Path) -> tuple[int, int, int, int] | None:
    """
    Threshold the ellipse PNG and return a tight axis-aligned bounding box.
    Returns (x1, y1, x2, y2) or None if the mask is empty.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Merge all contours into one bounding rect (handles fragmented ellipses)
    x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        x1 = min(x1, cx)
        y1 = min(y1, cy)
        x2 = max(x2, cx + cw)
        y2 = max(y2, cy + ch)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def _collect_defective(class_dir: Path) -> list[tuple[Path, Path]]:
    """
    Return [(image_path, mask_path)] for all defective images in Train + Test.

    Naming conventions differ between splits:
      Train/Label/  →  same filename as image   (e.g. 0576.PNG → Train/0576.PNG)
      Test/Label/   →  filename has _label suffix (e.g. 0002_label.PNG → Test/0002.PNG)
    """
    pairs = []
    for split in ("Train", "Test"):
        split_dir = class_dir / split
        label_dir = split_dir / "Label"
        if not split_dir.exists() or not label_dir.exists():
            continue
        for mask_path in sorted(label_dir.glob("*.PNG")):
            stem = mask_path.stem
            # Strip _label suffix if present (Test split)
            if stem.endswith("_label"):
                img_stem = stem[: -len("_label")]
            else:
                img_stem = stem
            img_path = split_dir / (img_stem + ".PNG")
            if img_path.exists():
                pairs.append((img_path, mask_path))
            else:
                print(f"  [WARN] mask {mask_path.name} has no matching image ({img_path.name})")
    return pairs


def process_dagm():
    for sub in ("images", "masks", "labels_yolo", "labels_coco"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # Class names: class1 … class10
    classes = [f"class{i}" for i in range(1, NUM_CLASSES + 1)]
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"DAGM classes: {classes}")

    coco: dict[str, COCOBuilder] = {
        cls: COCOBuilder(class_to_id[cls], cls) for cls in classes
    }
    manifest_path = init_manifest(OUTPUT_ROOT / "size_manifest.csv")

    instance_id = 0

    for class_idx in range(1, NUM_CLASSES + 1):
        cls = f"class{class_idx}"
        class_dir = DAGM_INPUT / f"Class{class_idx}"
        if not class_dir.exists():
            print(f"  [WARN] {class_dir} not found, skipping.")
            continue

        pairs = _collect_defective(class_dir)
        if not pairs:
            print(f"  [WARN] no defective images found in {class_dir}")
            continue

        for n, (img_path, mask_path) in enumerate(pairs, start=1):
            bbox = _ellipse_mask_to_bbox(mask_path)
            if bbox is None:
                print(f"  [SKIP] empty/unreadable mask: {mask_path}")
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            base = f"dagm_{cls}_{n:04d}"
            out_img_name  = f"{base}_defect.png"
            out_mask_name = f"{base}_mask.png"
            out_lbl_name  = f"{base}_bbs.txt"

            shutil.copy2(img_path,  OUTPUT_ROOT / "images" / out_img_name)
            shutil.copy2(mask_path, OUTPUT_ROOT / "masks"  / out_mask_name)

            cid = class_to_id[cls]
            size_cat = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)

            yolo_line = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, cid)
            (OUTPUT_ROOT / "labels_yolo" / out_lbl_name).write_text(yolo_line)

            coco[cls].add_image(n, out_img_name, img_w, img_h)
            coco[cls].add_annotation(n, x1, y1, x2, y2, size_cat)

            instance_id += 1
            append_manifest(manifest_path, out_img_name, instance_id, cls,
                            x1, y1, x2, y2, img_w, img_h)

    for cls, builder in coco.items():
        builder.save(OUTPUT_ROOT / "labels_coco" / f"{cls}.json")

    mapping_lines = [f"{cid}: {cls}" for cls, cid in sorted(class_to_id.items(), key=lambda x: x[1])]
    (OUTPUT_ROOT / "class_mapping.txt").write_text("\n".join(mapping_lines))

    print(f"DAGM done. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_dagm()
