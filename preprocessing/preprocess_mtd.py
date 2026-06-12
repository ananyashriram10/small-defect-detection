"""
MTD (Magnetic Tile Defect) preprocessing: binary mask PNGs → bounding boxes → YOLO + COCO.

Kaggle input:  /kaggle/input/magnetic-tile-surface-defects/
               MT_Blowhole/, MT_Break/, MT_Crack/, MT_Fray/, MT_Free/, MT_Uneven/
               Each has Imgs/ with pairs:  <name>.jpg  +  <name>_gt.png  (mask)

Images without a _gt.png mask are skipped (treated as normal/unlabeled).

Each connected component in the mask becomes one bounding box instance.

Output layout:
  processed/MTD/
    images/       mtd_{class}_{nnnn}_defect.jpg
    masks/        mtd_{class}_{nnnn}_mask.png
    labels_yolo/  mtd_{class}_{nnnn}_bbs.txt
    labels_coco/  {class_name}.json
    class_mapping.txt
    size_manifest.csv
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils import (
    COCOBuilder, append_manifest, bbox_to_yolo,
    init_manifest,
)

MTD_INPUT = Path("/kaggle/input/magnetic-tile-surface-defects")
OUTPUT_ROOT = Path("/kaggle/working/processed/MTD")

# Folder prefix → class name
CATEGORY_MAP = {
    "MT_Blowhole": "blowhole",
    "MT_Break":    "break",
    "MT_Crack":    "crack",
    "MT_Fray":     "fray",
    "MT_Free":     "free",
    "MT_Uneven":   "uneven",
}


def _mask_to_bboxes(mask_path: Path) -> list[tuple[int, int, int, int]]:
    """Return list of (x1, y1, x2, y2) for each connected component in the mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    # Binarize (threshold handles near-white pixels)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):  # skip background label 0
        x1 = int(stats[i, cv2.CC_STAT_LEFT])
        y1 = int(stats[i, cv2.CC_STAT_TOP])
        w  = int(stats[i, cv2.CC_STAT_WIDTH])
        h  = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w > 0 and h > 0:
            bboxes.append((x1, y1, x1 + w, y1 + h))
    return bboxes


def process_mtd():
    for sub in ("images", "masks", "labels_yolo", "labels_coco"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    classes = sorted(CATEGORY_MAP.values())
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"MTD classes: {classes}")

    coco: dict[str, COCOBuilder] = {
        cls: COCOBuilder(class_to_id[cls], cls) for cls in classes
    }
    manifest_path = init_manifest(OUTPUT_ROOT / "size_manifest.csv")

    class_counter: dict[str, int] = {}
    instance_id = 0

    for folder_name, cls in sorted(CATEGORY_MAP.items()):
        imgs_dir = MTD_INPUT / folder_name / "Imgs"
        if not imgs_dir.exists():
            print(f"  [WARN] {imgs_dir} not found, skipping.")
            continue

        class_counter.setdefault(cls, 0)
        img_id_counter = 0

        # Mask is the same stem as the image but with .png extension.
        # e.g. exp1_num_108719.jpg  →  exp1_num_108719.png
        for img_path in sorted(imgs_dir.glob("*.jpg")):
            mask_path = imgs_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                continue  # no mask → normal image, skip

            bboxes = _mask_to_bboxes(mask_path)
            if not bboxes:
                print(f"  [SKIP] empty mask: {mask_path.name}")
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            class_counter[cls] += 1
            n = class_counter[cls]
            img_id_counter += 1

            base = f"mtd_{cls}_{n:04d}"
            out_img_name  = f"{base}_defect.jpg"
            out_mask_name = f"{base}_mask.png"
            out_lbl_name  = f"{base}_bbs.txt"

            shutil.copy2(img_path,  OUTPUT_ROOT / "images" / out_img_name)
            shutil.copy2(mask_path, OUTPUT_ROOT / "masks"  / out_mask_name)

            yolo_lines = []
            cid = class_to_id[cls]

            for (x1, y1, x2, y2) in bboxes:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                yolo_lines.append(bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, cid))

                coco[cls].add_image(img_id_counter, out_img_name, img_w, img_h)
                coco[cls].add_annotation(img_id_counter, x1, y1, x2, y2,
                                          categorize_str(x2 - x1, y2 - y1, img_w, img_h))

                instance_id += 1
                append_manifest(manifest_path, out_img_name, instance_id, cls,
                                x1, y1, x2, y2, img_w, img_h)

            if yolo_lines:
                (OUTPUT_ROOT / "labels_yolo" / out_lbl_name).write_text("\n".join(yolo_lines))

    for cls, builder in coco.items():
        builder.save(OUTPUT_ROOT / "labels_coco" / f"{cls}.json")

    mapping_lines = [f"{cid}: {cls}" for cls, cid in sorted(class_to_id.items(), key=lambda x: x[1])]
    (OUTPUT_ROOT / "class_mapping.txt").write_text("\n".join(mapping_lines))

    print(f"MTD done. Output: {OUTPUT_ROOT}")


def categorize_str(bw, bh, iw, ih) -> str:
    from utils import categorize_defect
    return categorize_defect(bw, bh, iw, ih)


if __name__ == "__main__":
    process_mtd()
