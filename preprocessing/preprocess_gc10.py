"""
GC10-DET preprocessing: PASCAL VOC XML bounding boxes → YOLO + per-class COCO JSON.

Kaggle input:  /kaggle/input/gc10det/
               Images in numbered folders 1/ through 10/.
               All XML annotations in a separate top-level 'lable/' folder
               (dataset typo — it really is 'lable', not 'label').
               XML stem matches image stem, e.g. img_01_3402617700_00001.xml
               → image is in one of the numbered folders with the same stem + .jpg.

Output layout:
  processed/GC10-DET/
    images/          gc10_{class}_{nnnn}_defect.jpg
    labels_yolo/     gc10_{class}_{nnnn}_bbs.txt
    labels_coco/     {class_name}.json   (one file per defect class)
    class_mapping.txt
    size_manifest.csv
"""

import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image

from utils import (
    COCOBuilder, append_manifest, bbox_to_yolo,
    categorize_defect, init_manifest,
)

GC10_INPUT = Path("/kaggle/input/gc10det")
OUTPUT_ROOT = Path("/kaggle/working/processed/GC10-DET")


def _build_image_index(gc10_input: Path) -> dict[str, Path]:
    """
    Build a stem → image_path index by scanning all numbered class folders (1-10).
    This lets us look up an image by stem from any XML in the 'lable/' folder.
    """
    index = {}
    for folder in gc10_input.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    index[img_path.stem] = img_path
    return index


def _discover_classes(lable_dir: Path) -> list[str]:
    classes = set()
    for xml_path in lable_dir.glob("*.xml"):
        for obj in ET.parse(xml_path).getroot().findall("object"):
            classes.add(obj.find("name").text.strip())
    return sorted(classes)


def process_gc10():
    for sub in ("images", "labels_yolo", "labels_coco"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # XMLs live in 'lable/' (dataset's typo)
    lable_dir = GC10_INPUT / "lable"
    if not lable_dir.exists():
        raise FileNotFoundError(f"Expected annotation folder at {lable_dir}")

    image_index = _build_image_index(GC10_INPUT)
    print(f"GC10-DET: indexed {len(image_index)} images across class folders.")

    classes = _discover_classes(lable_dir)
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"GC10-DET: discovered {len(classes)} classes → {classes}")

    coco: dict[str, COCOBuilder] = {
        cls: COCOBuilder(class_to_id[cls], cls) for cls in classes
    }
    manifest_path = init_manifest(OUTPUT_ROOT / "size_manifest.csv")

    class_counter: dict[str, int] = defaultdict(int)
    instance_id = 0
    class_image_id: dict[str, dict[str, int]] = defaultdict(dict)

    for xml_path in sorted(lable_dir.glob("*.xml")):
        img_path = image_index.get(xml_path.stem)
        if img_path is None:
            print(f"  [SKIP] no image found for {xml_path.name}")
            continue

        root = ET.parse(xml_path).getroot()
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

        # File naming uses primary (first) class
        primary_cls = objects[0].find("name").text.strip()
        safe_cls = primary_cls.replace(" ", "_").lower()
        class_counter[primary_cls] += 1
        n = class_counter[primary_cls]

        base = f"gc10_{safe_cls}_{n:04d}"
        out_img_name = f"{base}_defect{img_path.suffix.lower()}"
        out_lbl_name = f"{base}_bbs.txt"

        shutil.copy2(img_path, OUTPUT_ROOT / "images" / out_img_name)

        yolo_lines = []

        for obj in objects:
            cls = obj.find("name").text.strip()
            bndbox = obj.find("bndbox")
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)
            # Clamp to image bounds
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
            if x2 <= x1 or y2 <= y1:
                continue

            cid = class_to_id[cls]
            yolo_lines.append(bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, cid))

            size_cat = categorize_defect(x2 - x1, y2 - y1, img_w, img_h)

            # Track image ID per class COCO file (same stem may appear once per class)
            stem = xml_path.stem
            if stem not in class_image_id[cls]:
                class_image_id[cls][stem] = len(class_image_id[cls]) + 1
            img_id = class_image_id[cls][stem]

            coco[cls].add_image(img_id, out_img_name, img_w, img_h)
            coco[cls].add_annotation(img_id, x1, y1, x2, y2, size_cat)

            instance_id += 1
            append_manifest(manifest_path, out_img_name, instance_id, cls,
                            x1, y1, x2, y2, img_w, img_h)

        if yolo_lines:
            (OUTPUT_ROOT / "labels_yolo" / out_lbl_name).write_text("\n".join(yolo_lines))

    # Save per-class COCO JSONs
    for cls, builder in coco.items():
        safe_cls = cls.replace(" ", "_").lower()
        builder.save(OUTPUT_ROOT / "labels_coco" / f"{safe_cls}.json")

    # Class mapping for reference
    mapping_lines = [f"{cid}: {cls}" for cls, cid in sorted(class_to_id.items(), key=lambda x: x[1])]
    (OUTPUT_ROOT / "class_mapping.txt").write_text("\n".join(mapping_lines))

    print(f"GC10-DET done. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_gc10()
