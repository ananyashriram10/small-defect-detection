# Small Defect Detection

This repository contains the dataset preparation, training setup, and experiment tracking plan for YOLO-based small industrial defect detection.

The current goal is to compare full-image YOLO baselines against tiled training and tiled inference, while keeping the dataset split, model family, logging, and evaluation rules consistent across runs.

## Dataset Summary

The dataset combines 7 industrial anomaly and defect datasets.

Defect size buckets are locked as:

| Bucket | Area Ratio Definition | Images |
|---|---:|---:|
| Small | `<1%` of image area | 3,035 |
| Medium | `1-5%` of image area | 5,741 |
| Large | `>=5%` of image area | 3,894 |
| Total | all defect images | 12,670 |

## Dataset Breakdown

| Dataset | Surface / Domain | Small | Medium | Large | Total | Defect Types |
|---|---|---:|---:|---:|---:|---|
| GC10-DET | Steel sheet | 322 | 622 | 1,348 | 2,292 | punching, weld line, crescent gap, water spot, oil spot, silk spot, inclusion, rolled pit, crease, waist fold |
| MTD | Magnetic tiles | 188 | 57 | 143 | 388 | blowhole, break, crack, fray, uneven |
| DAGM | Textured industrial surfaces | 593 | 1,095 | 412 | 2,100 | 10 synthetic texture defect classes |
| KolektorSDD2 | Electrical commutators / metal parts | 77 | 177 | 102 | 356 | surface scratches, cracks, spots |
| MPDD | Metal parts | 123 | 92 | 67 | 282 | surface defects per part type |
| Severstal | Industrial steel sheet | 1,618 | 3,656 | 1,821 | 7,095 | inclusions, patches, linear scratches, edge defects |
| VisA | Manufactured parts | 114 | 42 | 1 | 157 | anomalies per product category |

## Constants Across Runs

These settings should stay the same for all experiment runs unless the run is explicitly testing that setting.

| Category | Constant |
|---|---|
| W&B project | `smallDefectDetection` |
| Dataset version | fixed cleaned dataset used by the frontend dataset hub |
| Defect size buckets | small `<1%`, medium `1-5%`, large `>=5%` |
| Train/val/test split | fixed split reused for every run |
| Split strategy | stratified by dataset, defect size bucket, and defect type where available |
| Random seed | same seed for split generation and training |
| Model family | YOLOv8 |
| Base checkpoint | `yolov8n.pt` for the first ladder of runs |
| Epoch budget | same epoch count across comparable runs |
| Early stopping patience | same patience across comparable runs |
| Optimizer setting | same optimizer setting across comparable runs |
| Batch size rule | keep same when possible; only reduce if image size or tiling causes memory limits |
| Validation set | same validation set for every run |
| Test set | same held-out test set for final comparison |
| Confidence threshold policy | evaluate consistently across all runs |
| IoU/NMS policy | same default policy except tiled overlap/global NMS run |
| Logging | every run logged to W&B with matching run name |

## Metrics To Report

Every run should report the same metric set:

| Metric | Why It Matters |
|---|---|
| `mAP50` | standard detection quality at IoU 0.50 |
| `mAP50-95` | stricter overall detection quality |
| precision | false-positive control |
| recall | missed-defect control |
| small-defect recall | primary project metric |
| recall by size bucket | shows whether small defects improve without hurting medium/large |
| recall by dataset | catches dataset-specific failures |
| false positives per image | important for practical inspection workflows |
| inference time per original image | compares full-image vs tiled inference cost |

Primary comparison metric:

```text
small-defect recall on the fixed test set
```

## Current Runs

These are the runs we are working on first.

| Run | W&B Run Name | Experiment | What Changes |
|---|---|---|---|
| A | `A_yolov8n_full_640` | Full-image baseline | Train YOLOv8n on full images with `imgsz=640` |
| B | `B_yolov8n_full_1024` | Higher-resolution full image | Same as Run A, but with `imgsz=1024` |
| C | `C_yolov8n_tiled_640` | Tiled training | Train using `640x640` tiles without overlap |
| D | `D_yolov8n_tiled_overlap_global_nms` | Tiled inference with overlap | Use overlapping tiles during inference, map boxes back to original image coordinates, then apply global NMS |

## Later Runs

These should only start after Runs A-D have comparable results.

| Run | W&B Run Name | Experiment | What Changes |
|---|---|---|---|
| E | `E_yolov8n_augmented` | Careful augmentation | Add modest brightness, contrast, blur, rotation, and limited scaling |
| F | `F_yolov8n_small_sampling` | Small-defect-aware sampling | Oversample small-defect images or tiles while keeping dataset balance |
| G | `G_yolov8n_p2_head` | P2 small-object head | Add a finer prediction head only if data and inference changes are not enough |

## Run Notes

For a run to count, it must have:

- W&B run logged under `smallDefectDetection`
- exact command or config recorded
- dataset split version recorded
- validation metrics recorded
- final test metrics recorded
- sample predictions saved
- checkpoint path saved outside normal Git tracking if the file is large
