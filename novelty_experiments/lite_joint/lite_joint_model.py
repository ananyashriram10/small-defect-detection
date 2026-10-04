"""LiteJointModel: one shared Swin-Tiny backbone forward pass feeding two independent heads.
Detection taps the backbone's raw per-stage outputs (confirmed 2026-08-21 against the real
checkpoint -- see swin_fpn_neck.py) through a new FPN-PAN neck + TAL/DFL detect head.
Segmentation reuses Mask2Former's own existing pixel decoder + transformer decoder + class/mask
predictor completely unmodified -- the exact same reuse novelty_model.py already established
and proved correct. Deliberately no cross-task fusion module here (that's an optional later
stage, gated on this simpler version actually beating baselines first).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_fpn_neck import SwinPAFPNNeck
from lite_detection_head import DecoupledDetectHead
from tal_assigner import TaskAlignedAssigner, make_anchor_points
from dfl_loss import DetectionHeadLoss


class LiteJointModel(nn.Module):
    def __init__(self, mask2former_model, num_det_classes=1, reg_max=16, neck_ch=128,
                 in_channels=(96, 192, 384, 768), strides=(4, 8, 16, 32)):
        super().__init__()
        self.m2f = mask2former_model
        self.strides = strides
        self.neck = SwinPAFPNNeck(in_channels=in_channels, neck_ch=neck_ch)
        self.det_head = DecoupledDetectHead(in_ch=neck_ch, num_classes=num_det_classes, reg_max=reg_max)

    def forward(self, pixel_values, pixel_mask=None):
        if pixel_mask is None:
            pixel_mask = torch.ones(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3],
                                     device=pixel_values.device)

        base_out = self.m2f.model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_hidden_states=True)

        # ---- detection branch: raw Swin stages, never touched by segmentation's decoder ----
        stage_features = list(base_out.encoder_hidden_states)
        neck_out = self.neck(stage_features)
        det_cls_logits, det_reg_logits = self.det_head(neck_out)
        feature_shapes = [f.shape[-2:] for f in neck_out]
        anchor_points, anchor_strides = make_anchor_points(feature_shapes, self.strides, device=pixel_values.device)

        # ---- segmentation branch: identical to novelty_model.py's existing, proven reuse ----
        pixel_features = base_out.pixel_decoder_last_hidden_state
        seg_queries = base_out.transformer_decoder_last_hidden_state
        seg_queries_normed = self.m2f.model.transformer_module.decoder.layernorm(seg_queries)
        seg_class_logits = self.m2f.class_predictor(seg_queries_normed)
        mask_predictor = self.m2f.model.transformer_module.decoder.mask_predictor
        H, W = pixel_features.shape[-2:]
        seg_mask_logits, _ = mask_predictor(seg_queries_normed.transpose(0, 1), pixel_features,
                                             attention_mask_target_size=(H, W))

        return {
            "det_cls_logits": det_cls_logits,
            "det_reg_logits": det_reg_logits,
            "anchor_points": anchor_points,
            "anchor_strides": anchor_strides,
            "seg_class_logits": seg_class_logits,
            "seg_mask_logits": seg_mask_logits,
        }


def compute_detection_loss(det_cls_logits, det_reg_logits, anchor_points, anchor_strides,
                            gt_boxes_list, gt_classes_list, assigner, head_loss_fn, num_classes):
    """Wires TaskAlignedAssigner's targets into a classification (BCE against TAL's soft
    targets, all anchors) + DetectionHeadLoss (DFL+IoU, foreground anchors only) combined loss.
    Shared by this file's self-test and train_lightweight_runpod.py -- not duplicated."""
    pred_scores = det_cls_logits.sigmoid().detach()  # TAL's alignment metric uses the model's
    pred_boxes = head_loss_fn.integral(det_reg_logits).detach()  # own current predictions, but
    ax, ay = anchor_points[:, 0].view(1, -1), anchor_points[:, 1].view(1, -1)  # must not backprop
    ltrb = pred_boxes * anchor_strides.view(1, -1, 1)  # through the assignment process itself --
    pred_boxes_xyxy = torch.stack([ax - ltrb[..., 0], ay - ltrb[..., 1],  # only through the loss
                                    ax + ltrb[..., 2], ay + ltrb[..., 3]], dim=-1)  # given the targets.

    target_scores, target_boxes, fg_mask = assigner(
        anchor_points, pred_scores, pred_boxes_xyxy, gt_boxes_list, gt_classes_list, num_classes)

    cls_loss = F.binary_cross_entropy_with_logits(det_cls_logits, target_scores, reduction="mean")
    box_total, box_parts = head_loss_fn(det_reg_logits, anchor_points, anchor_strides, target_boxes, fg_mask)

    total = cls_loss + box_total
    return total, {"cls": cls_loss.detach(), "n_fg": int(fg_mask.sum().item()), **box_parts}


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation

    torch.manual_seed(0)

    DATASET_ROOT = Path("../../processed_output")
    img_path = DATASET_ROOT / "DAGM" / "large" / "images" / "dagm_class10_0012_defect.png"
    label_path = DATASET_ROOT / "DAGM" / "large" / "labels_yolo" / "dagm_class10_0012_bbs.txt"
    mask_path = DATASET_ROOT / "DAGM" / "large" / "masks" / "dagm_class10_0012_mask.png"

    IMG_SIZE = 640
    image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
    pixel_values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mask_tensor = (torch.from_numpy(np.asarray(mask).copy()) > 0).float()

    real_boxes_cxcywh, real_classes = [], []
    for line in label_path.read_text().strip().splitlines():
        _, cx, cy, w, h = line.split()
        real_boxes_cxcywh.append([float(cx), float(cy), float(w), float(h)])
        real_classes.append(0)  # single real class here (num_det_classes=1), unlike DefectFormer's 0=bg/1=defect convention
    print(f"Real image: {img_path.name}, real YOLO boxes: {real_boxes_cxcywh}")

    # YOLO format is normalized cxcywh -> convert to absolute-pixel xyxy for the TAL/DFL pipeline
    boxes_t = torch.tensor(real_boxes_cxcywh) * IMG_SIZE
    gt_boxes_xyxy = torch.stack([
        boxes_t[:, 0] - boxes_t[:, 2] / 2, boxes_t[:, 1] - boxes_t[:, 3] / 2,
        boxes_t[:, 0] + boxes_t[:, 2] / 2, boxes_t[:, 1] + boxes_t[:, 3] / 2,
    ], dim=-1)
    gt_classes_t = torch.tensor(real_classes, dtype=torch.long)

    print("\nLoading real checkpoint and building LiteJointModel...")
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-tiny-ade-semantic", num_labels=2, ignore_mismatched_sizes=True)
    model = LiteJointModel(m2f, num_det_classes=1, reg_max=16, neck_ch=128)
    model.train()

    outputs = model(pixel_values)
    print("det_cls_logits:", tuple(outputs["det_cls_logits"].shape))
    print("det_reg_logits:", tuple(outputs["det_reg_logits"].shape))
    print("seg_class_logits:", tuple(outputs["seg_class_logits"].shape))
    print("seg_mask_logits:", tuple(outputs["seg_mask_logits"].shape))
    print("\nConfirming output_hidden_states=True didn't disturb the fields novelty_model.py's")
    print("segmentation path already relies on -- shapes must match its own established pattern:")
    assert outputs["seg_class_logits"].shape[0] == 1 and outputs["seg_class_logits"].shape[-1] == 3
    assert outputs["seg_mask_logits"].shape[-2:] == (160, 160), "mask logits should be at pixel_decoder's stride-4 resolution, same as novelty_model.py's own segmentation output"
    print("OK: segmentation output shapes are exactly what novelty_model.py's existing, proven code path produces.")

    assigner = TaskAlignedAssigner(topk=10, alpha=1.0, beta=6.0)
    head_loss_fn = DetectionHeadLoss(reg_max=16)
    det_loss, det_parts = compute_detection_loss(
        outputs["det_cls_logits"], outputs["det_reg_logits"], outputs["anchor_points"], outputs["anchor_strides"],
        [gt_boxes_xyxy], [gt_classes_t], assigner, head_loss_fn, num_classes=1)
    print(f"\ndetection loss: {det_loss.item():.4f}, parts: {det_parts}")
    assert det_parts["n_fg"] > 0, "the real GT box on this real image should get at least one anchor assigned -- 0 would mean the assigner or anchor grid is broken"

    seg_loss_dict = m2f.criterion(
        masks_queries_logits=outputs["seg_mask_logits"], class_queries_logits=outputs["seg_class_logits"],
        mask_labels=[mask_tensor.unsqueeze(0)], class_labels=[torch.tensor([1])],
    )
    seg_loss = sum(seg_loss_dict.values())
    print(f"segmentation loss: {seg_loss.item():.4f}")

    total_loss = det_loss + seg_loss
    total_loss.backward()

    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f"\nFull model: {n_grad}/{n_total} params with nonzero grad "
          f"(not 100% expected -- Swin's stochastic depth randomly drops blocks per forward pass, "
          f"same documented caveat as novelty_model.py's own self-test)")
    assert n_grad > 0.85 * n_total, f"expected the vast majority of params to receive gradient, got {n_grad}/{n_total}"

    neck_head_params = list(model.neck.parameters()) + list(model.det_head.parameters())
    neck_head_grad = sum(1 for p in neck_head_params if p.grad is not None and p.grad.abs().sum() > 0)
    assert neck_head_grad == len(neck_head_params), "the new neck+head (no stochastic depth) must have 100% gradient coverage, no exceptions"

    print("\nOK: LiteJointModel verified end-to-end on a real image, real YOLO box, and a real mask --")
    print("detection loss (TAL assignment + DFL/IoU, new) and segmentation loss (Mask2Former's own")
    print("unmodified criterion) both compute correctly and backpropagate through the shared backbone")
    print("and both task-specific heads.")
