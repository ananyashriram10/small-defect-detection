"""
Run all 3 dataset preprocessing scripts in order.
Paste this into a Kaggle notebook cell and run, or execute directly.

Required Kaggle datasets to add before running:
  - alex000kim/gc10det
  - alex000kim/magnetic-tile-surface-defects
  - mhskjelvareid/dagm-2007-competition-dataset-opticalinspection

Install dependencies:
  !pip install opencv-python-headless pillow
"""

from preprocess_gc10 import process_gc10
from preprocess_mtd import process_mtd
from preprocess_dagm import process_dagm

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1/3 — GC10-DET")
    print("=" * 60)
    process_gc10()

    print()
    print("=" * 60)
    print("Step 2/3 — MTD (Magnetic Tile Defects)")
    print("=" * 60)
    process_mtd()

    print()
    print("=" * 60)
    print("Step 3/3 — DAGM")
    print("=" * 60)
    process_dagm()

    print()
    print("All done. Output is in /kaggle/working/processed/")
    print("  GC10-DET/  MTD/  DAGM/")
    print("Each contains: images/ masks/ labels_yolo/ labels_coco/ size_manifest.csv")
