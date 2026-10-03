# YOLO-SM baseline

This folder contains a PyTorch implementation of the architecture described in:

> Xuebin Yue and Lin Meng, "YOLO-SM: A Lightweight Single-Class Multi-Deformation Object Detection Network," IEEE TETCI, 2024.

The implementation is adapted for this project’s combined one-class industrial-defect dataset. It uses the paper’s main design ideas:

- DCMNet backbone with densely connected multi-scale dilated depthwise convolutions.
- SAM stereoscopic channel/spatial attention.
- MC max-pooling plus convolution downsampling.
- GMF neck with SPP, GSConv2D, GS bottlenecks, and weighted feature fusion.
- Decoupled anchor-free detection head with SimOTA assignment.
- BCE classification/objectness losses and CIoU box regression loss.

This is a research reproduction, not an official author release. The paper does not publish an official YOLO-SM code repository, and some layer widths/repeats are reconstructed from the paper’s architecture description and figure.

`SAM` here means the paper’s Stereoscopic Attention Mechanism; it is not Meta’s Segment Anything model. This implementation is intentionally independent of Ultralytics because YOLO-SM is a custom DCMNet/GMF detector rather than a YOLOv8/YOLOv11 model configuration.

## Run

The training script expects either:

1. A prepared YOLO dataset with `images/{train,val,test}` and `labels/{train,val,test}`, or
2. The project’s source layout with `DATASET_ROOT/<dataset>/<size>/{images,labels_yolo}`.

For a prepared dataset:

```powershell
$env:YOLO_DATASET_ROOT = "C:\path\to\run_a_yolo_dataset"
$env:DEVICE = "cuda"
python baselines/detection/yolosm/train_yolosm_runpod.py
```

For a Kaggle/RunPod-style environment:

```bash
export YOLO_DATASET_ROOT=/kaggle/input/run-a-yolo-dataset/run_a_yolo_dataset
export BASE_DIR=/kaggle/working/yolosm
export EPOCHS=300
python baselines/detection/yolosm/train_yolosm_runpod.py
```

Use `SMOKE=1` for a short structural/training check. The default run uses 300 epochs to match the paper’s budget. Set `WANDB_MODE=disabled` to disable Weights & Biases logging.

The project default is 640 pixels so the result is directly comparable with the other baselines. Set `IMG_SIZE=416` for the paper’s input resolution.

## Outputs

Outputs are written under `BASE_DIR/runs/<RUN_NAME>` and include:

- `best_model.pt`
- `training_history.csv`
- `evaluation_metrics.csv`
- `summary.csv`
- `config.json`

The test report contains overall metrics, small/medium/large metrics, and per-dataset metrics whenever dataset prefixes are available in filenames or a preparation manifest is present.
