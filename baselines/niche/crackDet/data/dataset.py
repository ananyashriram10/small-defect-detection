"""Oriented sub-crack detection dataset.

The paper's own datasets (ONPP, ORC, OCCSD) were built in-house and were
never released publicly, and this project has no oriented-box crack
annotations of its own yet (its existing defect data is pixel-mask /
axis-aligned-box only). So this dataset class defines an explicit,
documented JSON annotation schema instead of hardcoding a source:

    [
      {
        "image": "relative/or/absolute/path.png",
        "width": 512, "height": 512,
        "boxes": [
          {"cx": 123.4, "cy": 88.0, "h": 40.0, "w": 12.0, "theta_deg": 37.5, "label": 0},
          ...
        ]
      },
      ...
    ]

`cx, cy, h, w` are in image-pixel coordinates (h = the box's long side
along its own orientation, w = the short side, matching the paper's
five-parameter definition, Sec. 3.1); `theta_deg` in [0, 180). `label` is
optional (defaults to 0) and indexes into `heatmap`'s class channel, for
datasets with more than one oriented-object category.

Preprocessing intentionally matches this project's other baseline scripts
(train_segnext_t_runpod.py, PIDNet/GCNet's *_kaggle.ipynb) wherever the
task allows it: BILINEAR image resizing and the same ImageNet mean/std
normalization, applied the same way (float / 255, then subtract mean and
divide by std as (3,1,1) tensors after the channel-first permute -- not
just numerically equivalent, the literal same code shape).

One place it deliberately does NOT copy those scripts: they resize images
by an independent (sx, sy) stretch straight to (IMG_SIZE, IMG_SIZE), which
is fine for plain semantic segmentation (a resized pixel mask is still a
valid pixel mask under any stretch) but is NOT fine for oriented boxes --
under a non-uniform (sx != sy) stretch, a rotated rectangle warps into a
general parallelogram, not another rotated rectangle, and its "angle"
stops being well-defined. So this loader instead does a uniform-scale
("letterbox") resize -- same scale factor s on both axes, zero-padded to
fill the remaining square -- which rescales every box's (cx, cy, h, w) by
that single scalar s and leaves theta_deg completely unchanged, exactly
like every other oriented-detection codebase (MMRotate, DOTA devkit) has
to handle this same constraint.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .target_generator import CrackDetTargetGenerator, OrientedBox, collate_targets

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class OrientedCrackDataset(Dataset):
    def __init__(self, annotation_file: str, image_root: Optional[str] = None,
                 input_size: int = 512, stride: int = 4, num_classes: int = 1,
                 augment: bool = False):
        assert input_size % 32 == 0, "input_size must be a multiple of 32 (ReEDNet downsamples x32)"
        with open(annotation_file, "r") as f:
            self.records = json.load(f)
        self.image_root = image_root or os.path.dirname(os.path.abspath(annotation_file))
        self.input_size = input_size
        self.augment = augment
        self.target_gen = CrackDetTargetGenerator(stride=stride, num_classes=num_classes)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _letterbox_resize(self, img: Image.Image, boxes: List[OrientedBox]):
        """Uniform-scale resize (same interpolation as every other baseline in this
        project, PIL BILINEAR) + zero-pad to a square -- see module docstring for why
        this can't be the other baselines' independent-axis stretch-resize once boxes
        carry an angle. `scale` is applied identically to cx, cy, h, and w; theta_deg
        is untouched (a uniform scale never changes a rotated rectangle's angle).
        """
        orig_w, orig_h = img.size
        scale = self.input_size / max(orig_h, orig_w)
        new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = np.asarray(resized, dtype=np.uint8)

        scaled = []
        for b in boxes:
            cx, cy = b.cx * scale, b.cy * scale
            if cx < new_w and cy < new_h:
                scaled.append(OrientedBox(cx=cx, cy=cy, h=b.h * scale, w=b.w * scale,
                                           theta_deg=b.theta_deg, label=b.label))
        return canvas, scaled

    @staticmethod
    def _hflip(img: np.ndarray, boxes: List[OrientedBox]) -> List[OrientedBox]:
        w = img.shape[1]
        flipped = []
        for b in boxes:
            theta = (180.0 - b.theta_deg) % 180.0
            flipped.append(OrientedBox(cx=w - b.cx, cy=b.cy, h=b.h, w=b.w, theta_deg=theta,
                                        label=b.label))
        return flipped

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = rec["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        img = self._load_image(img_path)

        boxes = [OrientedBox(cx=b["cx"], cy=b["cy"], h=b["h"], w=b["w"],
                              theta_deg=b["theta_deg"] % 180.0, label=b.get("label", 0))
                 for b in rec.get("boxes", [])]

        img, boxes = self._letterbox_resize(img, boxes)

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            boxes = self._hflip(img, boxes)

        # Same pattern as train_segnext_t_runpod.py / PIDNet / GCNet: float, /255,
        # permute to channel-first, then subtract ImageNet mean and divide by std.
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD

        targets = self.target_gen((self.input_size, self.input_size), boxes)
        return img_t, targets


def crackdet_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, collate_targets(targets)
