"""QueryBoxJointModel: shared Swin-Tiny backbone -> two branches.

MAIN branch (kept at inference): Mask2Former's existing, unmodified pixel decoder + transformer
decoder + class_predictor + mask_predictor, PLUS a new QueryBoxHead reading the same per-query
representation. Box supervision rides on Mask2Former's OWN class+mask Hungarian matching (see
train_query_box_runpod.py for where those indices get extracted and used) -- the box head never
has to learn its own matching.

AUX branch (train-only, deleted at inference): the exact SwinPAFPNNeck + DecoupledDetectHead
already built and verified in swin_fpn_neck.py/lite_detection_head.py, fed from the backbone's
raw per-stage outputs. Its own TAL-assigned dense detection loss regularizes the shared backbone
during training; it costs nothing once training is done -- compute_aux=False skips it entirely,
not just its loss.
"""
import torch
import torch.nn as nn

from swin_fpn_neck import SwinPAFPNNeck
from lite_detection_head import DecoupledDetectHead
from tal_assigner import make_anchor_points
from query_box_head import QueryBoxHead


class QueryBoxJointModel(nn.Module):
    def __init__(self, mask2former_model, aux_num_det_classes=1, aux_reg_max=16, aux_neck_ch=128,
                 aux_in_channels=(96, 192, 384, 768), aux_strides=(4, 8, 16, 32)):
        super().__init__()
        self.m2f = mask2former_model
        hidden_dim = mask2former_model.config.hidden_dim
        self.query_box_head = QueryBoxHead(hidden_dim=hidden_dim)

        self.aux_strides = aux_strides
        self.aux_neck = SwinPAFPNNeck(in_channels=aux_in_channels, neck_ch=aux_neck_ch)
        self.aux_det_head = DecoupledDetectHead(in_ch=aux_neck_ch, num_classes=aux_num_det_classes, reg_max=aux_reg_max)

    def forward(self, pixel_values, pixel_mask=None, compute_aux=True):
        """compute_aux=True (training): both branches run. compute_aux=False (inference/deploy):
        the aux branch is never even computed, not just excluded from the loss -- matches
        'literally cut off the whole left branch' at deployment."""
        if pixel_mask is None:
            pixel_mask = torch.ones(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3],
                                     device=pixel_values.device)

        base_out = self.m2f.model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                                   output_hidden_states=compute_aux)

        # ---- main branch: identical to novelty_model.py's/lite_joint_model.py's existing, proven reuse ----
        pixel_features = base_out.pixel_decoder_last_hidden_state
        seg_queries = base_out.transformer_decoder_last_hidden_state
        seg_queries_normed = self.m2f.model.transformer_module.decoder.layernorm(seg_queries)
        seg_class_logits = self.m2f.class_predictor(seg_queries_normed)
        mask_predictor = self.m2f.model.transformer_module.decoder.mask_predictor
        H, W = pixel_features.shape[-2:]
        seg_mask_logits, _ = mask_predictor(seg_queries_normed.transpose(0, 1), pixel_features,
                                             attention_mask_target_size=(H, W))
        query_boxes = self.query_box_head(seg_queries_normed)

        out = {
            "seg_class_logits": seg_class_logits,
            "seg_mask_logits": seg_mask_logits,
            "query_boxes": query_boxes,
        }

        if not compute_aux:
            return out

        # ---- aux branch: train-only, exactly the existing dense detector, unmodified ----
        stage_features = list(base_out.encoder_hidden_states)
        neck_out = self.aux_neck(stage_features)
        aux_cls_logits, aux_reg_logits = self.aux_det_head(neck_out)
        feature_shapes = [f.shape[-2:] for f in neck_out]
        anchor_points, anchor_strides = make_anchor_points(feature_shapes, self.aux_strides, device=pixel_values.device)

        out.update({
            "aux_cls_logits": aux_cls_logits, "aux_reg_logits": aux_reg_logits,
            "anchor_points": anchor_points, "anchor_strides": anchor_strides,
        })
        return out


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation

    from query_box_head import query_box_loss, box_mask_consistency_loss
    from dfl_loss import DetectionHeadLoss
    from tal_assigner import TaskAlignedAssigner
    from lite_joint_model import compute_detection_loss

    torch.manual_seed(0)
    IMG_SIZE = 640
    DATASET_ROOT = Path("../../processed_output")
    img_path = DATASET_ROOT / "DAGM" / "large" / "images" / "dagm_class10_0012_defect.png"
    label_path = DATASET_ROOT / "DAGM" / "large" / "labels_yolo" / "dagm_class10_0012_bbs.txt"
    mask_path = DATASET_ROOT / "DAGM" / "large" / "masks" / "dagm_class10_0012_mask.png"

    image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
    pixel_values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mask_tensor = (torch.from_numpy(np.asarray(mask).copy()) > 0).float()

    real_boxes_cxcywh = []
    for line in label_path.read_text().strip().splitlines():
        _, cx, cy, w, h = line.split()
        real_boxes_cxcywh.append([float(cx), float(cy), float(w), float(h)])
    print(f"Real image: {img_path.name}, real YOLO boxes: {real_boxes_cxcywh}")

    # Mask2Former's matcher treats the whole image as ONE merged instance (class_labels=[1],
    # one binary mask) -- so the box target for that same one instance must also be ONE box:
    # the union/enclosing box of every real YOLO box on this image, in xyxy pixels then back to
    # normalized cxcywh. This keeps the box target consistent with what the mask target already
    # represents, rather than inventing a second, disagreeing convention.
    boxes_t = torch.tensor(real_boxes_cxcywh)
    x1 = (boxes_t[:, 0] - boxes_t[:, 2] / 2).min()
    y1 = (boxes_t[:, 1] - boxes_t[:, 3] / 2).min()
    x2 = (boxes_t[:, 0] + boxes_t[:, 2] / 2).max()
    y2 = (boxes_t[:, 1] + boxes_t[:, 3] / 2).max()
    union_box = torch.tensor([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]])
    print(f"Union GT box (normalized cxcywh, matching the one-merged-instance mask convention): {union_box.tolist()}")

    print("\nLoading real checkpoint and building QueryBoxJointModel...")
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-tiny-ade-semantic", num_labels=2, ignore_mismatched_sizes=True)
    model = QueryBoxJointModel(m2f)
    model.train()

    outputs = model(pixel_values, compute_aux=True)
    print("seg_class_logits:", tuple(outputs["seg_class_logits"].shape))
    print("seg_mask_logits:", tuple(outputs["seg_mask_logits"].shape))
    print("query_boxes:", tuple(outputs["query_boxes"].shape))
    assert outputs["query_boxes"].shape == (1, 100, 4)
    assert (outputs["query_boxes"] >= 0).all() and (outputs["query_boxes"] <= 1).all()

    print("\n--- Extracting Mask2Former's OWN class+mask matched indices (no second matcher) ---")
    mask_labels = [mask_tensor.unsqueeze(0)]
    class_labels = [torch.tensor([1])]
    matched_indices = m2f.criterion.matcher(
        outputs["seg_mask_logits"], outputs["seg_class_logits"], mask_labels, class_labels)
    print("matched_indices:", matched_indices)
    assert len(matched_indices) == 1 and matched_indices[0][0].numel() == 1, \
        "with exactly one GT instance per image (this project's existing convention), exactly one query should match"

    seg_loss_dict = m2f.criterion(
        masks_queries_logits=outputs["seg_mask_logits"], class_queries_logits=outputs["seg_class_logits"],
        mask_labels=mask_labels, class_labels=class_labels)
    seg_loss = sum(seg_loss_dict.values())
    print(f"seg_loss (unmodified Mask2Former criterion): {seg_loss.item():.4f}")

    qbox_loss, qbox_parts = query_box_loss(outputs["query_boxes"], [union_box], matched_indices)
    print(f"query_box_loss: {qbox_loss.item():.4f}, parts={qbox_parts}")

    mask_probs = outputs["seg_mask_logits"].sigmoid()
    cons_loss = box_mask_consistency_loss(outputs["query_boxes"], mask_probs, matched_indices)
    print(f"box_mask_consistency_loss: {cons_loss.item():.4f}")

    det_targets = [union_box]
    det_classes = [torch.zeros(1, dtype=torch.long)]
    x1p, y1p = (union_box[:, 0] - union_box[:, 2] / 2) * IMG_SIZE, (union_box[:, 1] - union_box[:, 3] / 2) * IMG_SIZE
    x2p, y2p = (union_box[:, 0] + union_box[:, 2] / 2) * IMG_SIZE, (union_box[:, 1] + union_box[:, 3] / 2) * IMG_SIZE
    gt_boxes_xyxy_pixel = [torch.stack([x1p, y1p, x2p, y2p], dim=-1)]
    assigner = TaskAlignedAssigner(topk=10, alpha=1.0, beta=6.0)
    head_loss_fn = DetectionHeadLoss(reg_max=16)
    aux_loss, aux_parts = compute_detection_loss(
        outputs["aux_cls_logits"], outputs["aux_reg_logits"], outputs["anchor_points"], outputs["anchor_strides"],
        gt_boxes_xyxy_pixel, det_classes, assigner, head_loss_fn, num_classes=1)
    print(f"aux_loss (dense, train-only): {aux_loss.item():.4f}, parts={aux_parts}")

    total_loss = seg_loss + 1.0 * qbox_loss + 1.0 * aux_loss + 0.5 * cons_loss
    total_loss.backward()
    print(f"\ntotal_loss: {total_loss.item():.4f}")

    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Full model: {n_grad}/{n_total} params with nonzero grad "
          f"(not 100% expected -- Swin's stochastic depth randomly drops blocks per forward pass)")
    assert n_grad > 0.85 * n_total

    qbox_params = list(model.query_box_head.parameters())
    qbox_grad = sum(1 for p in qbox_params if p.grad is not None and p.grad.abs().sum() > 0)
    assert qbox_grad == len(qbox_params), "the new query box head must have full gradient coverage"

    print("\n--- Inference-mode check: compute_aux=False must skip the aux branch entirely ---")
    model.eval()
    with torch.inference_mode():
        infer_out = model(pixel_values, compute_aux=False)
    assert "aux_cls_logits" not in infer_out, "aux branch outputs must not even be computed at inference"
    assert set(infer_out.keys()) == {"seg_class_logits", "seg_mask_logits", "query_boxes"}
    print("OK: inference-mode output has exactly the 3 deployed heads, no aux branch computed at all.")

    print("\nOK: QueryBoxJointModel verified end-to-end on a real image, real YOLO boxes, and a real")
    print("mask -- all 4 loss terms (seg, query-box on Mask2Former's own matched indices, dense aux,")
    print("box-mask consistency) compute correctly and backpropagate, and the aux branch is fully")
    print("skippable at inference as designed.")
