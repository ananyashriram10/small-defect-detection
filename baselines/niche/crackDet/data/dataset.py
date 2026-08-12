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

Images are expected to already be sliced to a fixed square `input_size`
(the paper slices all 3 of its datasets into 512x512 patches, Sec. 3.5);
this loader center-crops/pads instead of resizing so box geometry (and
therefore theta_deg) never needs to change.
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        return np.asarray(img, dtype=np.uint8)

    def _pad_or_crop(self, img: np.ndarray, boxes: List[OrientedBox]):
        h, w = img.shape[:2]
        s = self.input_size
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        h_use, w_use = min(h, s), min(w, s)
        canvas[:h_use, :w_use] = img[:h_use, :w_use]

        kept = []
        for b in boxes:
            if b.cx < w_use and b.cy < h_use:
                kept.append(b)
        return canvas, kept

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

        img, boxes = self._pad_or_crop(img, boxes)

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            boxes = self._hflip(img, boxes)

        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).contiguous()

        targets = self.target_gen((self.input_size, self.input_size), boxes)
        return img_t, targets


def crackdet_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, collate_targets(targets)
