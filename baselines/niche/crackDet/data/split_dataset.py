"""Split a sliced CrackDet dataset into train/val/test at the paper's own ratio,
stratified across source datasets -- same pattern as every other RunPod script
in this project (train_mask2former_runpod.py, train_segnext_t_runpod.py, etc.).

Sec. 4.1 gives the ratio: "we divide each dataset into the training,
validation, and test set, with the proportion of 8:1:1." That's the only
detail the paper gives -- but the paper's own ONPP/ORC/OCCSD are each a
single-source dataset split on its own, which is NOT our situation. This
project trains on a combined dataset pulled from multiple of
DAGM/GC10-DET/KolektorSDD2/MPDD/MTD/Severstal/VisA, exactly like every
other baseline here, so a plain unstratified random shuffle risks an
unlucky split -- e.g. one source dataset landing almost entirely in test.
So this stratifies by (dataset, size_bucket) the same way
train_mask2former_runpod.py's `by_stratum` does: group records by stratum,
shuffle within each group (seeded), take the paper's 80/10/10 from *each*
group, then concatenate and shuffle each final split. The 8:1:1 ratio is
the paper's; the stratification is this project's own established
convention, not the paper's -- CrackDet's paper never needed it because it
never combines datasets the way we are.

Input: a single annotations JSON in the schema `slice_dataset.py` produces
(list of {"image", "width", "height", "dataset", "size_bucket", "boxes"}
records, one per patch).

Output: train.json / val.json / test.json in --output-dir, each a subset of
the input records in the same schema -- OrientedCrackDataset loads these
directly. Images are NOT moved/duplicated; all three splits point at the
same images/ directory.

Usage:
    python split_dataset.py --input sliced/annotations.json --output-dir sliced/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test gets the remainder: 1 - TRAIN_RATIO - VAL_RATIO = 0.1
SEED = 42


def stratum_key(record: dict) -> str:
    return f"{record.get('dataset', 'unknown')}_{record.get('size_bucket', 'unknown')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="sliced annotations JSON to split")
    parser.add_argument("--output-dir", required=True, help="where to write train/val/test.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = json.load(f)

    by_stratum = defaultdict(list)
    for rec in records:
        by_stratum[stratum_key(rec)].append(rec)

    rng = Random(args.seed)
    train, val, test = [], [], []
    for _, group in sorted(by_stratum.items()):
        group = list(group)
        rng.shuffle(group)
        n_train = int(round(len(group) * TRAIN_RATIO))
        n_val = int(round(len(group) * VAL_RATIO))
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(split, f)

    def box_count(split):
        return sum(len(r.get("boxes", [])) for r in split)

    def print_stratum_counts(name, split):
        counts = Counter(stratum_key(r) for r in split)
        print(f"  {name}: total={len(split)} boxes={box_count(split)}")
        for stratum, count in sorted(counts.items()):
            print(f"    {stratum}: {count}")

    print(f"Total patches: {len(records)} across {len(by_stratum)} stratum/strata (seed={args.seed})")
    print_stratum_counts("train", train)
    print_stratum_counts("val", val)
    print_stratum_counts("test", test)
    print(f"Written to {output_dir / 'train.json'}, {output_dir / 'val.json'}, {output_dir / 'test.json'}")

    if any(k.endswith("_unknown") or k.startswith("unknown_") for k in by_stratum):
        print("WARNING: some records had no 'dataset'/'size_bucket' field (grouped under "
              "'unknown') -- set --input records' dataset/size_bucket if you want real "
              "stratification across your source datasets.")


if __name__ == "__main__":
    main()
