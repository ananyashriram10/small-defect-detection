# ============================================================
# Cell 1 — Install + Imports
# ============================================================
# !pip install opencv-python-headless pillow -q

import csv
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================================
# Cell 2 — CONFIG
# ============================================================

GC10_INPUT         = Path("/kaggle/input/gc10det")
MTD_INPUT          = Path("/kaggle/input/magnetic-tile-surface-defects")
DAGM_INPUT         = Path("/kaggle/input/dagm-2007-competition-dataset-opticalinspection")
KOLEKTORSDD2_INPUT = Path("/kaggle/input/kolektorsdd2")
SEVERSTAL_INPUT    = Path("/kaggle/input/severstal-steel-defect-detection")
OUTPUT_ROOT        = Path("/kaggle/working/processed")

SMALL_THRESHOLD = 0.01   # bbox_area / image_area < 1%  → small
LARGE_THRESHOLD = 0.05   # bbox_area / image_area ≥ 5%  → large

SIZE_CATS = ("small", "medium", "large")

MTD_CATEGORY_MAP = {
    "MT_Blowhole": "blowhole",
    "MT_Break":    "break",
    "MT_Crack":    "crack",
    "MT_Fray":     "fray",
    "MT_Free":     "free",
    "MT_Uneven":   "uneven",
}

SEVERSTAL_H, SEVERSTAL_W = 256, 1600

# ============================================================
# Cell 3 — Shared utilities
# ============================================================

SIZE_RANK = {"small": 0, "medium": 1, "large": 2}


def categorize_defect(bbox_w, bbox_h, img_w, img_h):
    ratio = (bbox_w * bbox_h) / (img_w * img_h)
    if ratio < SMALL_THRESHOLD:
        return "small"
    elif ratio < LARGE_THRESHOLD:
        return "medium"
    else:
        return "large"


def image_size_category(box_size_cats):
    """Largest-defect-wins: if any box is large → large, elif any medium → medium, else small."""
    return max(box_size_cats, key=lambda s: SIZE_RANK[s])


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, class_id):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def make_size_dirs(dataset_out: Path, subdirs=("images", "masks", "labels_yolo")):
    """Create small/ medium/ large/ subfolders each containing the given subdirs."""
    for size in SIZE_CATS:
        for sub in subdirs:
            (dataset_out / size / sub).mkdir(parents=True, exist_ok=True)


class COCOBuilder:
    def __init__(self, class_id, class_name):
        self.class_id   = class_id
        self.class_name = class_name
        self._images    = {}
        self._anns      = []
        self._ann_id    = 1

    def add_image(self, image_id, file_name, width, height):
        if image_id not in self._images:
            self._images[image_id] = {
                "id": image_id, "file_name": file_name,
                "width": int(width), "height": int(height),
            }

    def add_annotation(self, image_id, x1, y1, x2, y2, size_cat):
        bw, bh = int(x2 - x1), int(y2 - y1)
        self._anns.append({
            "id": self._ann_id, "image_id": image_id,
            "category_id": self.class_id,
            "bbox": [int(x1), int(y1), bw, bh],
            "area": bw * bh, "iscrowd": 0,
            "size_category": size_cat,
        })
        self._ann_id += 1

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coco = {
            "images":      list(self._images.values()),
            "annotations": self._anns,
            "categories":  [{"id": self.class_id, "name": self.class_name}],
        }
        path.write_text(json.dumps(coco, indent=2))


MANIFEST_HEADER = [
    "image_file", "size_folder", "instance_id", "class_name",
    "x1", "y1", "x2", "y2",
    "bbox_area", "image_area", "relative_area", "instance_size_category",
]


def init_manifest(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(MANIFEST_HEADER)
    return path


def append_manifest(path, image_file, size_folder, instance_id, class_name,
                    x1, y1, x2, y2, img_w, img_h):
    bbox_area  = (x2 - x1) * (y2 - y1)
    image_area = img_w * img_h
    rel_area   = bbox_area / image_area
    inst_cat   = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            image_file, size_folder, instance_id, class_name,
            int(x1), int(y1), int(x2), int(y2),
            int(bbox_area), int(image_area), f"{rel_area:.6f}", inst_cat,
        ])
    return inst_cat

# ============================================================
# Cell 4 — GC10-DET preprocessing
# ============================================================

def _gc10_build_image_index(gc10_input):
    index = {}
    for folder in gc10_input.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    index[img_path.stem] = img_path
    return index


def _gc10_discover_classes(lable_dir):
    classes = set()
    for xml_path in lable_dir.glob("*.xml"):
        for obj in ET.parse(xml_path).getroot().findall("object"):
            classes.add(obj.find("name").text.strip())
    return sorted(classes)


def _bb_to_mask(boxes, img_w, img_h):
    """Return a uint8 grayscale mask with filled white rectangles for each (x1,y1,x2,y2) box."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        mask[int(y1):int(y2), int(x1):int(x2)] = 255
    return mask


def process_gc10():
    out = OUTPUT_ROOT / "GC10-DET"
    make_size_dirs(out, subdirs=("images", "masks", "labels_yolo"))
    (out / "labels_coco").mkdir(parents=True, exist_ok=True)

    lable_dir = GC10_INPUT / "lable"
    if not lable_dir.exists():
        raise FileNotFoundError(f"Annotation folder not found: {lable_dir}")

    image_index    = _gc10_build_image_index(GC10_INPUT)
    classes        = _gc10_discover_classes(lable_dir)
    class_to_id    = {c: i for i, c in enumerate(classes)}
    coco           = {cls: COCOBuilder(class_to_id[cls], cls) for cls in classes}
    manifest_path  = init_manifest(out / "size_manifest.csv")
    class_counter  = defaultdict(int)
    class_image_id = defaultdict(dict)
    instance_id    = 0
    counts         = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})

    for xml_path in sorted(lable_dir.glob("*.xml")):
        img_path = image_index.get(xml_path.stem)
        if img_path is None:
            continue

        root    = ET.parse(xml_path).getroot()
        objects = root.findall("object")
        if not objects:
            continue

        size_node = root.find("size")
        if size_node is not None:
            img_w = int(size_node.find("width").text)
            img_h = int(size_node.find("height").text)
        else:
            with Image.open(img_path) as im:
                img_w, img_h = im.size

        # ── Collect all valid boxes first to determine image-level size folder ──
        parsed_boxes = []
        for obj in objects:
            cls    = obj.find("name").text.strip()
            bndbox = obj.find("bndbox")
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            size_cat = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)
            parsed_boxes.append((cls, x1, y1, x2, y2, size_cat))

        if not parsed_boxes:
            continue

        img_folder   = image_size_category([b[5] for b in parsed_boxes])
        primary_cls  = parsed_boxes[0][0]
        safe_cls     = primary_cls.replace(" ", "_").lower()
        class_counter[primary_cls] += 1
        n = class_counter[primary_cls]

        base          = f"gc10_{safe_cls}_{n:04d}"
        out_img_name  = f"{base}_defect{img_path.suffix.lower()}"
        out_mask_name = f"{base}_mask.png"
        out_lbl_name  = f"{base}_bbs.txt"

        shutil.copy2(img_path, out / img_folder / "images" / out_img_name)
        counts[primary_cls][img_folder] += 1

        # Generate filled-rectangle pixel mask from bounding boxes
        mask_arr = _bb_to_mask([(x1, y1, x2, y2) for (_, x1, y1, x2, y2, _) in parsed_boxes],
                                img_w, img_h)
        Image.fromarray(mask_arr).save(out / img_folder / "masks" / out_mask_name)

        yolo_lines = []
        for (cls, x1, y1, x2, y2, size_cat) in parsed_boxes:
            cid = class_to_id[cls]
            yolo_lines.append(bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, cid))

            stem = xml_path.stem
            if stem not in class_image_id[cls]:
                class_image_id[cls][stem] = len(class_image_id[cls]) + 1
            img_id = class_image_id[cls][stem]

            coco[cls].add_image(img_id, out_img_name, img_w, img_h)
            coco[cls].add_annotation(img_id, x1, y1, x2, y2, size_cat)

            instance_id += 1
            append_manifest(manifest_path, out_img_name, img_folder, instance_id,
                            cls, x1, y1, x2, y2, img_w, img_h)

        (out / img_folder / "labels_yolo" / out_lbl_name).write_text("\n".join(yolo_lines))

    for cls, builder in coco.items():
        safe_cls = cls.replace(" ", "_").lower()
        builder.save(out / "labels_coco" / f"{safe_cls}.json")

    mapping = [f"{cid}: {cls}" for cls, cid in sorted(class_to_id.items(), key=lambda x: x[1])]
    (out / "class_mapping.txt").write_text("\n".join(mapping))
    return counts


# ============================================================
# Cell 5 — MTD preprocessing
# ============================================================

def _mask_to_bboxes(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):
        x1 = int(stats[i, cv2.CC_STAT_LEFT])
        y1 = int(stats[i, cv2.CC_STAT_TOP])
        w  = int(stats[i, cv2.CC_STAT_WIDTH])
        h  = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w > 0 and h > 0:
            bboxes.append((x1, y1, x1 + w, y1 + h))
    return bboxes


def process_mtd():
    out = OUTPUT_ROOT / "MTD"
    make_size_dirs(out, subdirs=("images", "masks", "labels_yolo"))
    (out / "labels_coco").mkdir(parents=True, exist_ok=True)

    classes       = sorted(MTD_CATEGORY_MAP.values())
    class_to_id   = {c: i for i, c in enumerate(classes)}
    coco          = {cls: COCOBuilder(class_to_id[cls], cls) for cls in classes}
    manifest_path = init_manifest(out / "size_manifest.csv")
    class_counter = {}
    instance_id   = 0
    counts        = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})

    for folder_name, cls in sorted(MTD_CATEGORY_MAP.items()):
        imgs_dir = MTD_INPUT / folder_name / "Imgs"
        if not imgs_dir.exists():
            print(f"  [WARN] {imgs_dir} not found, skipping.")
            continue

        class_counter.setdefault(cls, 0)
        img_id_counter = 0

        for img_path in sorted(imgs_dir.glob("*.jpg")):
            mask_path = imgs_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                continue

            raw_bboxes = _mask_to_bboxes(mask_path)
            if not raw_bboxes:
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            # Collect valid boxes + per-instance size
            parsed_boxes = []
            for (x1, y1, x2, y2) in raw_bboxes:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                size_cat = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)
                parsed_boxes.append((x1, y1, x2, y2, size_cat))

            if not parsed_boxes:
                continue

            img_folder = image_size_category([b[4] for b in parsed_boxes])

            class_counter[cls] += 1
            n = class_counter[cls]
            img_id_counter += 1

            base          = f"mtd_{cls}_{n:04d}"
            out_img_name  = f"{base}_defect.jpg"
            out_mask_name = f"{base}_mask.png"
            out_lbl_name  = f"{base}_bbs.txt"

            shutil.copy2(img_path,  out / img_folder / "images" / out_img_name)
            shutil.copy2(mask_path, out / img_folder / "masks"  / out_mask_name)
            counts[cls][img_folder] += 1

            cid        = class_to_id[cls]
            yolo_lines = []

            for (x1, y1, x2, y2, size_cat) in parsed_boxes:
                yolo_lines.append(bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, cid))
                coco[cls].add_image(img_id_counter, out_img_name, img_w, img_h)
                coco[cls].add_annotation(img_id_counter, x1, y1, x2, y2, size_cat)

                instance_id += 1
                append_manifest(manifest_path, out_img_name, img_folder, instance_id,
                                cls, x1, y1, x2, y2, img_w, img_h)

            (out / img_folder / "labels_yolo" / out_lbl_name).write_text("\n".join(yolo_lines))

    for cls, builder in coco.items():
        builder.save(out / "labels_coco" / f"{cls}.json")

    mapping = [f"{cid}: {cls}" for cls, cid in sorted(class_to_id.items(), key=lambda x: x[1])]
    (out / "class_mapping.txt").write_text("\n".join(mapping))
    return counts


# ============================================================
# Cell 6 — DAGM preprocessing
# ============================================================

def _ellipse_to_bbox(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        x1 = min(x1, cx);  y1 = min(y1, cy)
        x2 = max(x2, cx + cw);  y2 = max(y2, cy + ch)
    return (int(x1), int(y1), int(x2), int(y2)) if x2 > x1 and y2 > y1 else None


def _dagm_collect_defective(class_dir):
    pairs = []
    for split in ("Train", "Test"):
        split_dir = class_dir / split
        label_dir = split_dir / "Label"
        if not split_dir.exists() or not label_dir.exists():
            continue
        for mask_path in sorted(label_dir.glob("*.PNG")):
            stem     = mask_path.stem
            img_stem = stem[: -len("_label")] if stem.endswith("_label") else stem
            img_path = split_dir / (img_stem + ".PNG")
            if img_path.exists():
                pairs.append((img_path, mask_path))
    return pairs


def process_dagm():
    out = OUTPUT_ROOT / "DAGM"
    make_size_dirs(out, subdirs=("images", "masks", "labels_yolo"))
    (out / "labels_coco").mkdir(parents=True, exist_ok=True)

    classes       = [f"class{i}" for i in range(1, 11)]
    class_to_id   = {c: i for i, c in enumerate(classes)}
    coco          = {cls: COCOBuilder(class_to_id[cls], cls) for cls in classes}
    manifest_path = init_manifest(out / "size_manifest.csv")
    instance_id   = 0
    counts        = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})

    for class_idx in range(1, 11):
        cls       = f"class{class_idx}"
        class_dir = DAGM_INPUT / f"Class{class_idx}"
        if not class_dir.exists():
            print(f"  [WARN] {class_dir} not found, skipping.")
            continue

        pairs = _dagm_collect_defective(class_dir)
        for n, (img_path, mask_path) in enumerate(pairs, start=1):
            bbox = _ellipse_to_bbox(mask_path)
            if bbox is None:
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            size_cat   = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)
            img_folder = size_cat   # DAGM has one box per image

            base          = f"dagm_{cls}_{n:04d}"
            out_img_name  = f"{base}_defect.png"
            out_mask_name = f"{base}_mask.png"
            out_lbl_name  = f"{base}_bbs.txt"

            shutil.copy2(img_path,  out / img_folder / "images" / out_img_name)
            shutil.copy2(mask_path, out / img_folder / "masks"  / out_mask_name)
            counts[cls][img_folder] += 1

            cid = class_to_id[cls]
            (out / img_folder / "labels_yolo" / out_lbl_name).write_text(
                bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, cid)
            )
            coco[cls].add_image(n, out_img_name, img_w, img_h)
            coco[cls].add_annotation(n, x1, y1, x2, y2, size_cat)

            instance_id += 1
            append_manifest(manifest_path, out_img_name, img_folder, instance_id,
                            cls, x1, y1, x2, y2, img_w, img_h)

    for cls, builder in coco.items():
        builder.save(out / "labels_coco" / f"{cls}.json")

    mapping = [f"{cid}: {cls}" for cls, cid in sorted(class_to_id.items(), key=lambda x: x[1])]
    (out / "class_mapping.txt").write_text("\n".join(mapping))
    return counts


# ============================================================
# Cell 5b — KolektorSDD2 preprocessing
# ============================================================
# Dataset layout:
#   kolektorsdd2/
#     train/  and  test/
#       positive/   (defective)
#         part0/  part1/  …
#           image.jpg  +  image_label.bmp   (binary mask, white = defect)
#       negative/   (defect-free, skipped)

def process_kolektorsdd2():
    out = OUTPUT_ROOT / "KolektorSDD2"
    make_size_dirs(out, subdirs=("images", "masks", "labels_yolo"))
    (out / "labels_coco").mkdir(parents=True, exist_ok=True)

    cls        = "surface_defect"
    class_to_id = {cls: 0}
    coco       = {cls: COCOBuilder(0, cls)}
    manifest_path = init_manifest(out / "size_manifest.csv")
    counts     = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
    instance_id = 0
    n = 0

    for split in ("train", "test"):
        pos_dir = KOLEKTORSDD2_INPUT / split / "positive"
        if not pos_dir.exists():
            print(f"  [WARN] {pos_dir} not found, skipping.")
            continue

        for img_path in sorted(pos_dir.rglob("*.jpg")):
            # mask is same stem + _label.bmp
            mask_src = img_path.parent / (img_path.stem + "_label.bmp")
            if not mask_src.exists():
                continue

            mask_arr = np.array(Image.open(mask_src).convert("L"))
            # threshold — mask pixels > 0 are defect
            _, binary = cv2.threshold(mask_arr, 0, 255, cv2.THRESH_BINARY)

            # Get bounding boxes from connected components
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            bboxes = []
            for i in range(1, num_labels):
                x1 = int(stats[i, cv2.CC_STAT_LEFT])
                y1 = int(stats[i, cv2.CC_STAT_TOP])
                w  = int(stats[i, cv2.CC_STAT_WIDTH])
                h  = int(stats[i, cv2.CC_STAT_HEIGHT])
                if w > 0 and h > 0:
                    bboxes.append((x1, y1, x1 + w, y1 + h))

            if not bboxes:
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            size_cats  = [categorize_defect(x2-x1, y2-y1, img_w, img_h) for (x1,y1,x2,y2) in bboxes]
            img_folder = image_size_category(size_cats)

            n += 1
            base          = f"kolektorsdd2_{cls}_{n:04d}"
            out_img_name  = f"{base}_defect.jpg"
            out_mask_name = f"{base}_mask.png"
            out_lbl_name  = f"{base}_bbs.txt"

            shutil.copy2(img_path, out / img_folder / "images" / out_img_name)
            Image.fromarray(binary).save(out / img_folder / "masks" / out_mask_name)
            counts[cls][img_folder] += 1

            yolo_lines = []
            for idx, (x1, y1, x2, y2) in enumerate(bboxes):
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                size_cat = size_cats[idx]
                yolo_lines.append(bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, 0))
                coco[cls].add_image(n, out_img_name, img_w, img_h)
                coco[cls].add_annotation(n, x1, y1, x2, y2, size_cat)
                instance_id += 1
                append_manifest(manifest_path, out_img_name, img_folder, instance_id,
                                cls, x1, y1, x2, y2, img_w, img_h)

            if yolo_lines:
                (out / img_folder / "labels_yolo" / out_lbl_name).write_text("\n".join(yolo_lines))

    coco[cls].save(out / "labels_coco" / f"{cls}.json")
    (out / "class_mapping.txt").write_text("0: surface_defect")
    return counts


# ============================================================
# Cell 5c — Severstal preprocessing
# ============================================================
# Dataset layout:
#   severstal-steel-defect-detection/
#     train_images/   {ImageId}.jpg   (1600×256)
#     train.csv       ImageId, ClassId, EncodedPixels  (RLE, column-major, 1-indexed)
#   Classes 1-4 (defect types); rows with no EncodedPixels are defect-free.

def _decode_rle(rle_str, height, width):
    """Decode Kaggle column-major RLE into a uint8 binary mask."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if not isinstance(rle_str, str) or rle_str.strip() == "":
        return mask.reshape(height, width)
    tokens = list(map(int, rle_str.split()))
    for start, length in zip(tokens[0::2], tokens[1::2]):
        start -= 1  # 1-indexed → 0-indexed
        mask[start: start + length] = 255
    # RLE is column-major (Fortran order)
    return mask.reshape(height, width, order="F")


def process_severstal():
    out = OUTPUT_ROOT / "Severstal"
    make_size_dirs(out, subdirs=("images", "masks", "labels_yolo"))
    (out / "labels_coco").mkdir(parents=True, exist_ok=True)

    csv_path  = SEVERSTAL_INPUT / "train.csv"
    img_dir   = SEVERSTAL_INPUT / "train_images"
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {csv_path}")

    # Group rows by ImageId
    rows_by_image = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rle = row.get("EncodedPixels", "").strip()
            if rle and rle != "nan":
                rows_by_image[row["ImageId"]].append(
                    (int(row["ClassId"]), rle)
                )

    classes     = [f"defect{i}" for i in range(1, 5)]
    class_to_id = {c: i for i, c in enumerate(classes)}
    coco        = {cls: COCOBuilder(class_to_id[cls], cls) for cls in classes}
    manifest_path = init_manifest(out / "size_manifest.csv")
    counts      = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
    instance_id = 0
    img_counters = defaultdict(int)

    for image_id, defect_rows in sorted(rows_by_image.items()):
        img_path = img_dir / image_id
        if not img_path.exists():
            continue

        img_w, img_h = SEVERSTAL_W, SEVERSTAL_H

        # Merge all class masks; also collect per-class bboxes
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        all_bboxes    = []  # (cls, x1, y1, x2, y2, size_cat)

        for class_id, rle in defect_rows:
            cls      = f"defect{class_id}"
            cls_mask = _decode_rle(rle, img_h, img_w)
            combined_mask = np.maximum(combined_mask, cls_mask)

            # Find tight BB from this class mask
            _, binary = cv2.threshold(cls_mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            x1, y1, x2, y2 = img_w, img_h, 0, 0
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                x1 = min(x1, cx);      y1 = min(y1, cy)
                x2 = max(x2, cx + cw); y2 = max(y2, cy + ch)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            size_cat = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)
            all_bboxes.append((cls, x1, y1, x2, y2, size_cat))

        if not all_bboxes:
            continue

        img_folder   = image_size_category([b[5] for b in all_bboxes])
        primary_cls  = all_bboxes[0][0]
        img_counters[primary_cls] += 1
        n = img_counters[primary_cls]

        base          = f"severstal_{primary_cls}_{n:04d}"
        out_img_name  = f"{base}_defect.jpg"
        out_mask_name = f"{base}_mask.png"
        out_lbl_name  = f"{base}_bbs.txt"

        shutil.copy2(img_path, out / img_folder / "images" / out_img_name)
        Image.fromarray(combined_mask).save(out / img_folder / "masks" / out_mask_name)
        counts[primary_cls][img_folder] += 1

        yolo_lines = []
        for (cls, x1, y1, x2, y2, size_cat) in all_bboxes:
            cid = class_to_id[cls]
            yolo_lines.append(bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, cid))
            img_id = img_counters[cls]
            coco[cls].add_image(img_id, out_img_name, img_w, img_h)
            coco[cls].add_annotation(img_id, x1, y1, x2, y2, size_cat)
            instance_id += 1
            append_manifest(manifest_path, out_img_name, img_folder, instance_id,
                            cls, x1, y1, x2, y2, img_w, img_h)

        if yolo_lines:
            (out / img_folder / "labels_yolo" / out_lbl_name).write_text("\n".join(yolo_lines))

    for cls, builder in coco.items():
        builder.save(out / "labels_coco" / f"{cls}.json")

    mapping = [f"{class_to_id[c]}: {c}" for c in classes]
    (out / "class_mapping.txt").write_text("\n".join(mapping))
    return counts


# ============================================================
# Cell 7 — Count summary printer
# ============================================================

def print_counts(dataset_name, counts):
    total = sum(sum(sc.values()) for sc in counts.values())
    print(f"\n{'='*62}")
    print(f"  {dataset_name}  —  {total} defective images total")
    print(f"{'='*62}")
    print(f"  {'Class':<25} {'Total':>6}  {'Small':>6} {'Medium':>7} {'Large':>6}")
    print(f"  {'-'*25} {'-'*6}  {'-'*6} {'-'*7} {'-'*6}")
    grand = {"small": 0, "medium": 0, "large": 0}
    for cls in sorted(counts):
        sc  = counts[cls]
        tot = sum(sc.values())
        print(f"  {cls:<25} {tot:>6}  {sc['small']:>6} {sc['medium']:>7} {sc['large']:>6}")
        for k in grand:
            grand[k] += sc[k]
    print(f"  {'TOTAL':<25} {total:>6}  {grand['small']:>6} {grand['medium']:>7} {grand['large']:>6}")
    print()


# ============================================================
# Cell 8 — Visualisation
# ============================================================

def _read_yolo_boxes(yolo_path, img_w, img_h):
    boxes = []
    if not Path(yolo_path).exists():
        return boxes
    for line in Path(yolo_path).read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        boxes.append((cid, x1, y1, x2, y2))
    return boxes


COLORS = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759",
    "#00C7BE", "#30B0C7", "#007AFF", "#5856D6",
    "#AF52DE", "#FF2D55",
]
BG_COLOR     = "#0d0d0d"
PANEL_COLOR  = "#1a1a1a"
TEXT_COLOR   = "#f0f0f0"
ACCENT_COLOR = "#00C7BE"

SIZE_ACCENT = {"small": "#34C759", "medium": "#FF9500", "large": "#FF3B30"}


def _parse_class_from_filename(img_name):
    parts = img_name.split("_")
    return "_".join(parts[1:-2]) if len(parts) > 3 else "unknown"


def visualise_sample(dataset_name, n_per_size=2, save=True):
    """
    Sample n_per_size images from each of small/ medium/ large/ folders
    and display: defective image | pixel mask | bounding boxes with labels.
    Saves a PNG to /kaggle/working/.
    """
    out = OUTPUT_ROOT / dataset_name

    id_to_cls = {}
    mapping_path = out / "class_mapping.txt"
    if mapping_path.exists():
        for line in mapping_path.read_text().splitlines():
            if ": " in line:
                cid_str, cname = line.split(": ", 1)
                id_to_cls[int(cid_str)] = cname

    size_lookup = defaultdict(list)
    manifest_path = out / "size_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                size_lookup[row["image_file"]].append(row["instance_size_category"])

    # Build sample list: [(size_folder, img_path), ...]
    sample_list = []
    for size in SIZE_CATS:
        img_dir    = out / size / "images"
        all_images = sorted(img_dir.glob("*_defect.*")) if img_dir.exists() else []
        chosen     = random.sample(all_images, min(n_per_size, len(all_images)))
        for p in chosen:
            sample_list.append((size, p))

    if not sample_list:
        print(f"No processed images found under {out}")
        return

    n_rows = len(sample_list)
    fig    = plt.figure(figsize=(17, 5.5 * n_rows), facecolor=BG_COLOR)
    fig.suptitle(
        f"{dataset_name}  ·  Sample Visualisation  (small / medium / large)",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )

    col_titles = ["Defective Image", "Pixel Mask", "Bounding Boxes"]

    for row_idx, (size_folder, img_path) in enumerate(sample_list):
        axes = [fig.add_subplot(n_rows, 3, row_idx * 3 + col + 1) for col in range(3)]

        if row_idx == 0:
            for ax, title in zip(axes, col_titles):
                ax.set_title(title, fontsize=11, fontweight="bold",
                             color=ACCENT_COLOR, pad=10)

        img_pil      = Image.open(img_path).convert("RGB")
        img_arr      = np.array(img_pil)
        img_h, img_w = img_arr.shape[:2]
        cls_name     = _parse_class_from_filename(img_path.name)
        size_color   = SIZE_ACCENT[size_folder]

        # ── Col 1: defective image ───────────────────────────────
        ax_img = axes[0]
        ax_img.set_facecolor(PANEL_COLOR)
        ax_img.imshow(img_arr)
        ax_img.set_xlabel(
            f"{img_path.name}\n{img_w}×{img_h}px  |  class: {cls_name}",
            fontsize=7.5, color=TEXT_COLOR, labelpad=5,
        )
        ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_img.spines.values():
            spine.set_edgecolor(size_color)
            spine.set_linewidth(2)

        # ── Col 2: pixel mask ────────────────────────────────────
        ax_mask = axes[1]
        ax_mask.set_facecolor(PANEL_COLOR)
        mask_stem = _replace_last(img_path.stem, "_defect", "_mask")
        mask_path = out / size_folder / "masks" / (mask_stem + ".png")
        if mask_path.exists():
            mask_arr  = np.array(Image.open(mask_path).convert("L"))
            ax_mask.imshow(mask_arr, cmap="inferno", vmin=0, vmax=255)
            defect_px = int(np.sum(mask_arr > 127))
            rel_pct   = 100 * defect_px / (img_w * img_h)
            ax_mask.set_xlabel(
                f"defect pixels: {defect_px:,}  ({rel_pct:.2f}% of image)",
                fontsize=7.5, color=TEXT_COLOR, labelpad=5,
            )
        else:
            ax_mask.text(0.5, 0.5, "No pixel mask\n(bounding-box only dataset)",
                         ha="center", va="center", fontsize=9,
                         color="#888888", transform=ax_mask.transAxes)
            ax_mask.set_xlabel("—", fontsize=7.5, color="#555555", labelpad=5)
        ax_mask.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_mask.spines.values():
            spine.set_edgecolor(size_color)
            spine.set_linewidth(2)

        # ── Col 3: bounding boxes ────────────────────────────────
        ax_bb = axes[2]
        ax_bb.set_facecolor(PANEL_COLOR)
        ax_bb.imshow(img_arr)

        lbl_stem  = _replace_last(img_path.stem, "_defect", "_bbs")
        lbl_path  = out / size_folder / "labels_yolo" / (lbl_stem + ".txt")
        boxes     = _read_yolo_boxes(lbl_path, img_w, img_h)
        size_cats = size_lookup.get(img_path.name, [])

        for bb_idx, (cid, x1, y1, x2, y2) in enumerate(boxes):
            bw        = x2 - x1
            bh        = y2 - y1
            color     = COLORS[cid % len(COLORS)]
            inst_cat  = size_cats[bb_idx] if bb_idx < len(size_cats) else "?"
            cls_label = id_to_cls.get(cid, str(cid))

            ax_bb.add_patch(mpatches.Rectangle(
                (x1, y1), bw, bh,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.12,
            ))
            ax_bb.add_patch(mpatches.Rectangle(
                (x1, y1), bw, bh,
                linewidth=2, edgecolor=color, facecolor="none",
            ))
            label_y = y1 - 6 if y1 > 18 else y2 + 6
            ax_bb.text(
                x1 + 2, label_y,
                f"{cls_label}  [{inst_cat}]",
                fontsize=7, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=color, alpha=0.85, linewidth=0),
            )
            ax_bb.text(
                x1 + 2, y2 + 9,
                f"x={x1} y={y1}  w={bw} h={bh}",
                fontsize=6.5, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.65, linewidth=0),
            )

        n_inst = len(boxes)
        ax_bb.set_xlabel(
            f"{n_inst} instance{'s' if n_inst != 1 else ''}  "
            f"| size: {', '.join(size_cats) if size_cats else '—'}",
            fontsize=7.5, color=TEXT_COLOR, labelpad=5,
        )
        ax_bb.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_bb.spines.values():
            spine.set_edgecolor(size_color)
            spine.set_linewidth(2)

        # Row label showing which size bucket this sample came from
        axes[0].set_ylabel(
            size_folder.upper(),
            fontsize=11, fontweight="bold", color=size_color,
            rotation=90, labelpad=10,
        )

    plt.tight_layout(rect=[0, 0, 1, 1])

    if save:
        save_path = Path("/kaggle/working") / f"viz_{dataset_name.lower()}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        print(f"  Saved → {save_path}")

    plt.show()


# ============================================================
# Cell 9 — RUN EVERYTHING
# ============================================================

random.seed(42)

print("Processing GC10-DET ...")
gc10_counts = process_gc10()
print_counts("GC10-DET", gc10_counts)

print("Processing MTD ...")
mtd_counts = process_mtd()
print_counts("MTD", mtd_counts)

print("Processing DAGM ...")
dagm_counts = process_dagm()
print_counts("DAGM", dagm_counts)

print("Processing KolektorSDD2 ...")
kolektorsdd2_counts = process_kolektorsdd2()
print_counts("KolektorSDD2", kolektorsdd2_counts)

print("Processing Severstal ...")
severstal_counts = process_severstal()
print_counts("Severstal", severstal_counts)

all_total = (sum(sum(sc.values()) for sc in gc10_counts.values()) +
             sum(sum(sc.values()) for sc in mtd_counts.values()) +
             sum(sum(sc.values()) for sc in dagm_counts.values()) +
             sum(sum(sc.values()) for sc in kolektorsdd2_counts.values()) +
             sum(sum(sc.values()) for sc in severstal_counts.values()))
print(f"\n{'='*62}")
print(f"  GRAND TOTAL across all 5 datasets: {all_total} defective images")
print(f"{'='*62}\n")

# ============================================================
# Cell 10 — VISUALISE (2 samples per size bucket per dataset)
# ============================================================

print("Visualising GC10-DET ...")
visualise_sample("GC10-DET", n_per_size=2)

print("Visualising MTD ...")
visualise_sample("MTD", n_per_size=2)

print("Visualising DAGM ...")
visualise_sample("DAGM", n_per_size=2)

print("Visualising KolektorSDD2 ...")
visualise_sample("KolektorSDD2", n_per_size=2)

print("Visualising Severstal ...")
visualise_sample("Severstal", n_per_size=2)
