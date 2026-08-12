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

Preprocessing follows the paper exactly, not this project's other baseline
scripts. The other baselines (SegNeXt/PIDNet/GCNet) resize raw images to a
fixed size at *training* time via an independent (sx, sy) stretch, because
that's harmless for a plain pixel mask. CrackDet's paper never does that at
all -- Sec. 3.5 states the datasets are built by slicing full-resolution
source images into non-overlapping 512x512 patches ONCE, ahead of time, as
a dataset-construction step ("we directly slice these images into 512x512
pixels"), not resizing per-batch inside training. This loader mirrors that
exactly: it expects every image to already be precisely `input_size` x
`input_size` and raises a hard error otherwise, instead of silently
resizing/padding/cropping it into shape. Resizing here would in any case
be lossy for oriented boxes in a way slicing is not -- a plain crop only
ever translates a box (angle and side lengths are untouched), whereas any
resize with sx != sy warps a rotated rectangle into a non-rectangular
parallelogram, corrupting theta_deg. Use `slice_dataset.py` in this same
directory to turn full-resolution images + box annotations into
`input_size`-sized patches the way the paper's own ONPP/ORC/OCCSD were
built, before pointing this loader at them.
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

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if img.size != (self.input_size, self.input_size):
            raise ValueError(
                f"{path}: image is {img.size[0]}x{img.size[1]}, expected exactly "
                f"{self.input_size}x{self.input_size}. This dataset expects pre-sliced "
                f"patches (paper Sec. 3.5), not arbitrary-sized images -- run "
                f"slice_dataset.py first to produce {self.input_size}x{self.input_size} "
                f"patches from your full-resolution images + annotations."
            )
        return np.asarray(img, dtype=np.uint8)

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

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            boxes = self._hflip(img, boxes)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD

        targets = self.target_gen((self.input_size, self.input_size), boxes)
        return img_t, targets


def crackdet_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, collate_targets(targets)
