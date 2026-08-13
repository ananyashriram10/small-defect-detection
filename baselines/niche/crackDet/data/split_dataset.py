"""Split a sliced CrackDet dataset into train/val/test at the paper's own ratio.

Sec. 4.1: "we divide each dataset into the training, validation, and test
set, with the proportion of 8:1:1." That's the only detail given -- no
stratification variable, no seed, nothing about how box density per patch
factors in. So this is a plain random shuffle-and-split at 80/10/10, seeded
for reproducibility. That's a real gap in what the paper specifies, not
something to pretend is more precise than it is: if patches vary a lot in
how many sub-cracks they contain, a plain random split can still end up
imbalanced across splits by chance. Flagging it here rather than adding
stratification the paper never asked for.

Input: a single annotations JSON in the schema `dataset.py` / `slice_dataset.py`
produce (list of {"image", "width", "height", "boxes"} records, one per
patch) -- typically slice_dataset.py's own output.json.

Output: train.json / val.json / test.json in --output-dir, each a subset of
the input records in the same schema, so `OrientedCrackDataset` can load
them directly (paths are copied through unchanged -- images are NOT moved
or duplicated, all three splits still point at the same images/ directory).

Usage:
    python split_dataset.py --input sliced/annotations.json --output-dir sliced/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test gets the remainder, so it's exactly 1 - TRAIN_RATIO - VAL_RATIO = 0.1
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="sliced annotations JSON to split")
    parser.add_argument("--output-dir", required=True, help="where to write train/val/test.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = json.load(f)

    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(split, f)

    def box_count(split):
        return sum(len(r.get("boxes", [])) for r in split)

    print(f"Total patches: {n} (seed={args.seed})")
    print(f"  train: {len(train)} patches, {box_count(train)} boxes -> {output_dir / 'train.json'}")
    print(f"  val:   {len(val)} patches, {box_count(val)} boxes -> {output_dir / 'val.json'}")
    print(f"  test:  {len(test)} patches, {box_count(test)} boxes -> {output_dir / 'test.json'}")


if __name__ == "__main__":
    main()
