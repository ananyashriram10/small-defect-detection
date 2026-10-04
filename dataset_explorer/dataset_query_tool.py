#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from metadata import (
        CLASS_TRANSLATIONS,
        DATASET_ORDER,
        LABEL_STRUCTURE,
        SIZE_ORDER,
        SURFACE_TYPES,
    )
except ImportError:
    from dataset_explorer.metadata import (
        CLASS_TRANSLATIONS,
        DATASET_ORDER,
        LABEL_STRUCTURE,
        SIZE_ORDER,
        SURFACE_TYPES,
    )


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(path):
    path = Path(path)
    if not path.exists():
        return 0
    image_dir = path / "images"
    root = image_dir if image_dir.exists() else path
    return sum(1 for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_class_mapping(dataset_dir):
    mapping_path = dataset_dir / "class_mapping.txt"
    if not mapping_path.exists():
        return []

    classes = []
    for line in mapping_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            _, name = line.split(":", 1)
            classes.append(name.strip())
        else:
            classes.append(line)
    return classes


def unique_image_counts(df, group_cols):
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["image_count"])
    return (
        df.groupby(group_cols)["image_file"]
        .nunique()
        .reset_index(name="image_count")
    )


def build_summary(processed_root):
    processed_root = Path(processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root does not exist: {processed_root}")

    dataset_rows = []
    class_rows = []

    dataset_dirs = [
        p for p in processed_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    dataset_dirs = sorted(
        dataset_dirs,
        key=lambda p: DATASET_ORDER.index(p.name) if p.name in DATASET_ORDER else 999,
    )

    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        manifest_path = dataset_dir / "size_manifest.csv"
        if not manifest_path.exists():
            continue

        df = pd.read_csv(manifest_path)
        required = {"image_file", "size_folder", "class_name"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")

        surface = SURFACE_TYPES.get(dataset, "unknown")
        label_type = LABEL_STRUCTURE.get(dataset, "unknown")
        class_map = CLASS_TRANSLATIONS.get(dataset, {})
        listed_classes = sorted(set(read_class_mapping(dataset_dir)).union(df["class_name"].dropna().astype(str)))

        size_counts = (
            unique_image_counts(df, ["size_folder"])
            .set_index("size_folder")["image_count"]
            .to_dict()
        )
        dataset_row = {
            "dataset": dataset,
            "surface_type": surface,
            "label_type": label_type,
            "num_classes": len(listed_classes),
            "small": int(size_counts.get("small", 0)),
            "medium": int(size_counts.get("medium", 0)),
            "large": int(size_counts.get("large", 0)),
        }
        dataset_row["total"] = dataset_row["small"] + dataset_row["medium"] + dataset_row["large"]
        dataset_row["classes"] = ", ".join(listed_classes)
        dataset_rows.append(dataset_row)

        grouped = unique_image_counts(df, ["class_name", "size_folder"])
        pivot = (
            grouped.pivot(index="class_name", columns="size_folder", values="image_count")
            .fillna(0)
            .astype(int)
        )

        for size in SIZE_ORDER:
            if size not in pivot.columns:
                pivot[size] = 0

        for listed_class in listed_classes:
            counts = pivot.loc[listed_class].to_dict() if listed_class in pivot.index else {}
            row = {
                "dataset": dataset,
                "surface_type": surface,
                "listed_class": listed_class,
                "translated_class": class_map.get(listed_class, "unmapped, review needed"),
                "label_type": label_type,
                "small": int(counts.get("small", 0)),
                "medium": int(counts.get("medium", 0)),
                "large": int(counts.get("large", 0)),
            }
            row["total"] = row["small"] + row["medium"] + row["large"]
            class_rows.append(row)

    dataset_summary = pd.DataFrame(dataset_rows)
    class_summary = pd.DataFrame(class_rows)

    if dataset_summary.empty:
        raise RuntimeError(f"No dataset summaries could be built from {processed_root}")

    aggregate = {
        "datasets": int(dataset_summary["dataset"].nunique()),
        "small": int(dataset_summary["small"].sum()),
        "medium": int(dataset_summary["medium"].sum()),
        "large": int(dataset_summary["large"].sum()),
    }
    aggregate["total"] = aggregate["small"] + aggregate["medium"] + aggregate["large"]

    return dataset_summary, class_summary, aggregate


def print_table(df, columns=None):
    view = df if columns is None else df[columns]
    print(view.to_string(index=False))


def export_outputs(dataset_summary, class_summary, aggregate, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    class_summary.to_csv(output_dir / "dataset_class_size_summary.csv", index=False)

    payload = {
        "aggregate": aggregate,
        "dataset_summary": dataset_summary.to_dict(orient="records"),
        "class_size_summary": class_summary.to_dict(orient="records"),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(payload, indent=2))

    print("Saved:", output_dir / "dataset_summary.csv")
    print("Saved:", output_dir / "dataset_class_size_summary.csv")
    print("Saved:", output_dir / "dataset_summary.json")


def build_parser():
    parser = argparse.ArgumentParser(description="Query the small defect processed dataset.")
    parser.add_argument(
        "command",
        choices=["datasets", "surfaces", "categories", "counts", "aggregate", "export"],
        help="Question to answer.",
    )
    parser.add_argument("--processed-root", default="processed_output")
    parser.add_argument("--output-dir", default="dataset_explorer/outputs")
    parser.add_argument("--dataset", help="Filter by dataset name.")
    parser.add_argument("--class-name", help="Filter by listed class name.")
    parser.add_argument("--label-type", help="Filter by label type.")
    return parser


def main():
    args = build_parser().parse_args()
    dataset_summary, class_summary, aggregate = build_summary(args.processed_root)

    if args.dataset:
        dataset_summary = dataset_summary[dataset_summary["dataset"] == args.dataset]
        class_summary = class_summary[class_summary["dataset"] == args.dataset]

    if args.class_name:
        class_summary = class_summary[class_summary["listed_class"] == args.class_name]

    if args.label_type:
        dataset_summary = dataset_summary[dataset_summary["label_type"] == args.label_type]
        class_summary = class_summary[class_summary["label_type"] == args.label_type]

    if args.command == "datasets":
        print_table(dataset_summary, ["dataset", "total", "small", "medium", "large", "num_classes"])
    elif args.command == "surfaces":
        print_table(dataset_summary, ["dataset", "surface_type", "label_type"])
    elif args.command == "categories":
        print_table(class_summary, ["dataset", "listed_class", "translated_class", "label_type"])
    elif args.command == "counts":
        print_table(
            class_summary,
            ["dataset", "listed_class", "translated_class", "small", "medium", "large", "total"],
        )
    elif args.command == "aggregate":
        print(json.dumps(aggregate, indent=2))
    elif args.command == "export":
        export_outputs(dataset_summary, class_summary, aggregate, args.output_dir)


if __name__ == "__main__":
    main()
