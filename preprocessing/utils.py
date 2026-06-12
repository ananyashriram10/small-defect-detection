"""Shared utilities for defect dataset preprocessing."""

import csv
import json
from pathlib import Path


SMALL_THRESHOLD = 0.01  # < 1% of image area
LARGE_THRESHOLD = 0.05  # >= 5% of image area


def categorize_defect(bbox_w, bbox_h, img_w, img_h):
    ratio = (bbox_w * bbox_h) / (img_w * img_h)
    if ratio < SMALL_THRESHOLD:
        return "small"
    elif ratio < LARGE_THRESHOLD:
        return "medium"
    else:
        return "large"


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, class_id):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


class COCOBuilder:
    """Builds a per-class COCO JSON with a custom size_category field per annotation."""

    def __init__(self, class_id, class_name):
        self.class_id = class_id
        self.class_name = class_name
        self._images = {}   # image_id -> dict
        self._anns = []
        self._ann_id = 1

    def add_image(self, image_id, file_name, width, height):
        if image_id not in self._images:
            self._images[image_id] = {
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
            }

    def add_annotation(self, image_id, x1, y1, x2, y2, size_cat):
        bw, bh = int(x2 - x1), int(y2 - y1)
        self._anns.append({
            "id": self._ann_id,
            "image_id": image_id,
            "category_id": self.class_id,
            "bbox": [int(x1), int(y1), bw, bh],
            "area": bw * bh,
            "iscrowd": 0,
            "size_category": size_cat,
        })
        self._ann_id += 1

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coco = {
            "images": list(self._images.values()),
            "annotations": self._anns,
            "categories": [{"id": self.class_id, "name": self.class_name}],
        }
        path.write_text(json.dumps(coco, indent=2))
        print(f"  Saved COCO JSON → {path}  ({len(self._anns)} annotations)")


MANIFEST_HEADER = [
    "image_file", "instance_id", "class_name",
    "x1", "y1", "x2", "y2",
    "bbox_area", "image_area", "relative_area", "size_category",
]


def init_manifest(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(MANIFEST_HEADER)
    return path


def append_manifest(path, image_file, instance_id, class_name,
                    x1, y1, x2, y2, img_w, img_h):
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = img_w * img_h
    rel_area = bbox_area / image_area
    size_cat = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            image_file, instance_id, class_name,
            int(x1), int(y1), int(x2), int(y2),
            int(bbox_area), int(image_area),
            f"{rel_area:.6f}", size_cat,
        ])
    return size_cat
