"""Replicate the paper's own dataset-construction step (Sec. 3.5): slice full-resolution
source images into fixed-size patches, ahead of training.

For all three of its own datasets, the paper describes exactly this and only this:
    ONPP:  "we directly slice these images into 512x512 pixels, constructing a
            final dataset with 3,104 samples."
    ORC:   "Similar to ONPP, we slice these images into 512x512 pixels,
            constructing a final road pavement dataset with 1,303 samples."
    OCCSD: "We construct an oriented sub-crack detection dataset with 1,875
            samples by slicing these images into 512x512 pixels and
            relabeling them."

No resizing anywhere -- patches are exact pixel crops of the source image, so
a box's angle and side lengths are copied through completely unchanged; only
its (cx, cy) needs to shift into patch-local coordinates. That's the whole
reason `dataset.py`'s loader is allowed to assume every image is already
exactly `input_size` x `input_size` and treat anything else as an error: in
the paper's own pipeline, that's a dataset-construction-time invariant, not
something the training-time loader should be resizing/padding to fix up.

One thing the paper does NOT specify precisely, and this script has to make
an explicit choice about: how patches with zero annotated sub-cracks are
handled. Exhaustive non-overlapping tiling of images at the paper's stated
source resolutions (e.g. ONPP: 200 images at 7360x4912) produces far more
512x512 tiles than the paper's final reported sample counts (3,104/1,303/
1,875), and cracks are visually sparse across a large image -- so the
authors' own pipeline must be dropping most background-only tiles somewhere,
without spelling out the exact rule. This script defaults to the same
behavior (`--keep-empty` off: drop patches with 0 boxes) as the closest
match to their reported counts being far smaller than an exhaustive tiling
would produce, but it is NOT a verbatim reproduction of their unstated
filtering rule -- flagged here rather than presented as more certain than
it is.

Input JSON schema (full-resolution source images, box coords in full-image
pixel space):
    [
      {"image": "raw/img1.png",
       "boxes": [{"cx": 4012.5, "cy": 2884.0, "h": 120.0, "w": 18.0, "theta_deg": 63.2, "label": 0}, ...]},
      ...
    ]

Output: `<output_dir>/images/<stem>_r{row}_c{col}.png` patches, plus an
annotation JSON in the exact schema `dataset.py`/`OrientedCrackDataset`
expects (box coords translated to patch-local, h/w/theta_deg untouched).

Usage:
    python slice_dataset.py --input raw_annotations.json --image-root /path/to/raw/images \
        --output-dir /path/to/sliced --patch-size 512
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def slice_image(image_path: Path, boxes: list, patch_size: int, keep_empty: bool):
    """Non-overlapping raster-order tiling; any partial edge tile is dropped
    (the paper's "slice into 512x512 pixels" language implies whole patches,
    not padded partial ones -- not explicitly stated either way).

    A box is assigned to the tile containing its center, with coordinates
    translated to that tile's local frame. h, w, theta_deg are copied through
    unchanged -- a crop is a pure translation, so nothing about box shape or
    orientation needs to (or should) change.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    n_cols, n_rows = w // patch_size, h // patch_size

    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            x0, y0 = col * patch_size, row * patch_size
            x1, y1 = x0 + patch_size, y0 + patch_size

            patch_boxes = []
            for b in boxes:
                if x0 <= b["cx"] < x1 and y0 <= b["cy"] < y1:
                    patch_boxes.append({
                        "cx": b["cx"] - x0, "cy": b["cy"] - y0,
                        "h": b["h"], "w": b["w"],
                        "theta_deg": b["theta_deg"] % 180.0,
                        "label": b.get("label", 0),
                    })

            if not patch_boxes and not keep_empty:
                continue

            patch_img = img.crop((x0, y0, x1, y1))
            patches.append((row, col, patch_img, patch_boxes))
    return patches


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="raw annotation JSON (full-res images)")
    parser.add_argument("--image-root", default=None,
                         help="directory raw 'image' paths are relative to (default: input file's dir)")
    parser.add_argument("--output-dir", required=True, help="where to write sliced patches + annotations")
    parser.add_argument("--patch-size", type=int, default=512,
                         help="paper uses 512 for ONPP/ORC/OCCSD, 768 for HRSC2016 (default: 512)")
    parser.add_argument("--keep-empty", action="store_true",
                         help="keep patches with zero boxes (default: drop them, see module docstring)")
    args = parser.parse_args()

    input_path = Path(args.input)
    image_root = Path(args.image_root) if args.image_root else input_path.resolve().parent
    output_dir = Path(args.output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f:
        records = json.load(f)

    sliced_records = []
    total_patches = total_boxes = total_empty_dropped = 0

    for rec in records:
        img_path = Path(rec["image"])
        if not img_path.is_absolute():
            img_path = image_root / img_path
        boxes = rec.get("boxes", [])
        stem = img_path.stem

        patches = slice_image(img_path, boxes, args.patch_size, args.keep_empty)
        for row, col, patch_img, patch_boxes in patches:
            out_name = f"{stem}_r{row}_c{col}.png"
            patch_img.save(images_out / out_name)
            sliced_records.append({
                "image": f"images/{out_name}",
                "width": args.patch_size, "height": args.patch_size,
                "boxes": patch_boxes,
            })
            total_patches += 1
            total_boxes += len(patch_boxes)

        w, h = Image.open(img_path).size
        n_tiles = (w // args.patch_size) * (h // args.patch_size)
        total_empty_dropped += n_tiles - len(patches)

    ann_out_path = output_dir / "annotations.json"
    with open(ann_out_path, "w") as f:
        json.dump(sliced_records, f)

    print(f"Sliced {len(records)} source image(s) -> {total_patches} patch(es), "
          f"{total_boxes} box(es) total.")
    if not args.keep_empty:
        print(f"Dropped {total_empty_dropped} empty (0-box) patch(es) -- pass --keep-empty to keep them.")
    print(f"Patches: {images_out}")
    print(f"Annotations: {ann_out_path}")


if __name__ == "__main__":
    main()
