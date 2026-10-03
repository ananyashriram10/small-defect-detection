"""Dependency-light one-class detection metrics for YOLO-SM."""

from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from yolosm_losses import box_iou


REPORT_CONFIDENCE = float(os.environ.get("REPORT_CONFIDENCE", "0.5"))


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.3, max_det: int = 300) -> Tensor:
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0 and len(keep) < max_det:
        idx = int(order[0])
        keep.append(idx)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[idx:idx + 1], boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _ap(scores: list[float], tp: list[bool], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp_arr = np.asarray(tp, dtype=np.float64)[order]
    fp_arr = 1.0 - tp_arr
    recall = np.cumsum(tp_arr) / n_gt
    precision = np.cumsum(tp_arr) / np.maximum(np.cumsum(tp_arr) + np.cumsum(fp_arr), 1e-9)
    for idx in range(len(precision) - 2, -1, -1):
        precision[idx] = max(precision[idx], precision[idx + 1])
    return float(sum(precision[np.searchsorted(recall, r, side="left")] if np.searchsorted(recall, r, side="left") < len(precision) else 0.0 for r in np.linspace(0, 1, 101)) / 101)


class DetectionAccumulator:
    thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, pred_boxes: Tensor, pred_scores: Tensor, gt_boxes: Tensor, dataset: str, size: str) -> None:
        self.records.append({
            "pred_boxes": pred_boxes.detach().cpu(),
            "pred_scores": pred_scores.detach().cpu(),
            "gt_boxes": gt_boxes.detach().cpu(),
            "dataset": dataset,
            "size": size,
        })

    def compute(self, selector=None) -> dict[str, float]:
        records = [r for r in self.records if selector is None or selector(r)]
        n_gt = sum(len(r["gt_boxes"]) for r in records)
        detections = {float(t): [] for t in self.thresholds}
        for threshold in self.thresholds:
            for record in records:
                boxes, scores, gt = record["pred_boxes"], record["pred_scores"], record["gt_boxes"]
                order = scores.argsort(descending=True)
                ious = box_iou(boxes, gt) if len(gt) else torch.zeros((len(boxes), 0))
                matched = set()
                for pred_idx in order.tolist():
                    is_tp = False
                    if len(gt):
                        best_gt = int(torch.argmax(ious[pred_idx]).item())
                        if float(ious[pred_idx, best_gt]) >= float(threshold) and best_gt not in matched:
                            matched.add(best_gt)
                            is_tp = True
                    detections[float(threshold)].append((float(scores[pred_idx]), is_tp))
        ap = {t: _ap([s for s, _ in values], [tp for _, tp in values], n_gt) for t, values in detections.items()}
        tp50 = sum(tp for score, tp in detections[0.5] if score >= REPORT_CONFIDENCE)
        fp50 = sum(not tp for score, tp in detections[0.5] if score >= REPORT_CONFIDENCE)
        return {
            "mAP50": ap[0.5],
            "mAP50_95": float(np.nanmean(list(ap.values()))) if ap else 0.0,
            "precision": tp50 / max(tp50 + fp50, 1),
            "recall": tp50 / max(n_gt, 1),
            "images": len(records),
            "ground_truth_boxes": n_gt,
        }


def summarize_accumulator(acc: DetectionAccumulator) -> list[dict[str, float | str]]:
    rows = [{"split": "overall", **acc.compute()}]
    for size in ("small", "medium", "large"):
        rows.append({"split": size, **acc.compute(lambda r, s=size: r["size"] == s)})
    datasets = sorted({str(r["dataset"]) for r in acc.records})
    for dataset in datasets:
        rows.append({"split": f"dataset:{dataset}", **acc.compute(lambda r, d=dataset: r["dataset"] == d)})
    return rows
