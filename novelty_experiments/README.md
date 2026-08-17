# DefectFormer (v1)

Joint detection + segmentation architecture proposed as this project's novel contribution, built on top of the 7+7 baseline comparison in `baselines/`.

## Architecture

**Backbone**: Mask2Former (Swin-Tiny, `facebook/mask2former-swin-tiny-ade-semantic`), chosen from a real benchmark across the segmentation baselines in this project — won on every metric, especially Recall_Small (0.714 vs SegNeXt's next-best 0.565), the metric this project is centered on.

**Contribution 1 — cross-task attention.** A second, parallel set of detection queries (`DetectionQueryDecoder`, a standard non-masked transformer decoder — Mask2Former's own masked-attention mechanism has no analog for box regression, so it isn't reused) runs alongside Mask2Former's native segmentation queries, with bidirectional cross-attention (`CrossTaskAttention`) between the two sets before either commits to a final prediction. Segmentation reuses Mask2Former's own `class_predictor`/`mask_predictor` on the cross-attended queries, so segmentation benefits from the cross-attention too, not just detection. Detection loss is a from-scratch DETR-style setup: Hungarian matching (classification + L1 + GIoU cost) plus the matching CE/L1/GIoU loss (`detection_loss.py`).

**Contribution 2 — prompt-grounded auxiliary task (training-only).** Contrastive alignment between cropped defect regions and real text descriptions (`defect_prompts_v2.json`, 27,230 real geometrically-derived instances). Text encoder is trained from scratch (small transformer, not a pretrained LLM) — the real corpus has only 196 unique words across all 571,830 description strings. Shares only the backbone with detection/segmentation, not the cross-attention; its gradients reach the shared backbone by being summed into the same backward pass, not through any query-level interaction. Dropped entirely at inference.

Code layout: `prompt_pair_dataset.py` (aux-task data), `det_seg_cross_attention.py` (detection decoder + cross-attention + detection head), `prompt_alignment_head.py` (aux-task model), `detection_loss.py` (Hungarian matcher + DETR loss), `detection_metrics.py` (single-class COCO-style mAP, size-stratified), `novelty_model.py` (combined model). Training script: `../train_novelty_runpod.py` (RunPod, same scaffolding as this project's other `train_*_runpod.py` scripts).

## Training

```
export WANDB_API_KEY=<key>
export DATASET_ROOT=/path/to/processed_output
export PROMPTS_JSON=/path/to/defect_prompts_v2.json
nohup python -u train_novelty_runpod.py > train.log 2>&1 &
```

50 epochs max (patience 15), batch=4, AdamW (backbone lr=1e-5, head lr=1e-4). bf16 autocast wraps forward passes only; detection/segmentation loss and both Hungarian matchers (Mask2Former's own for masks, the new one for boxes) are forced to fp32 explicitly. fp16 AMP is avoided on purpose — a different DETR-style detector in this project (D-FINE) produced NaN box predictions under plain fp16, and bf16 keeps fp32's exponent range specifically to avoid that failure class.

## v1 results (test set, 1920 images, best checkpoint = epoch 45/50)

Column note: `Dice`/`IoU`/`Recall`/`Precision` (overall and per size bucket) are all **segmentation** metrics. `mAP50`/`mAP50_95` are the only **detection** metrics — this model reports both since it does both tasks jointly, and the two shouldn't be conflated (e.g. `Recall_Small` below is segmentation recall on the small-image test bucket, not detection recall).

| Metric | Overall | Small | Medium | Large |
|---|---|---|---|---|
| Segmentation Dice | 0.7505 | 0.5083 | 0.6639 | 0.7841 |
| Segmentation IoU | 0.6006 | 0.3408 | 0.4969 | 0.6448 |
| Segmentation Recall | 0.7888 | 0.6587 | 0.7350 | 0.8069 |
| Detection mAP50 | 0.0020 | 0.0004 | 0.0025 | 0.0030 |

Inference: 54.8 ms/image. W&B run: https://wandb.ai/ananyashriram10-manipal/smallDefectDetection/runs/gnlv3xhk

**Segmentation works.** Dice/IoU are solid and consistent with the cross-attention design goal of segmentation benefiting from the joint setup.

**Detection does not work yet.** mAP50 is effectively zero across every bucket, including small — the metric this whole project is centered on — after the full 50-epoch budget. Two explanations are on the table, not yet distinguished:

1. The detection decoder is a standard, non-deformable DETR-style decoder — a family well documented as needing hundreds of epochs to converge (the original DETR paper needed ~500). This may simply be too early.
2. Detection loss plateaued at ~4.4x segmentation's loss for roughly the second half of training and never broke from that, and the W&B mAP50 curve across all 50 epochs is noisy around a near-zero floor rather than a slow-but-rising trend. Both observations fit a real bug or a gradient imbalance (segmentation's already-well-tuned loss dominating the shared backbone) better than they fit slow-but-working convergence.

Next diagnostic step, not yet done: inspect actual predicted boxes from the epoch-45 checkpoint against ground truth on real test images. Roughly-correct-but-low-confidence boxes would favor explanation 1; degenerate or near-identical boxes across queries would favor explanation 2.
