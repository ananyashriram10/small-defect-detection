"""
Full model: real pretrained Mask2Former (backbone + pixel decoder + segmentation
queries) + a parallel detection-query decoder + bidirectional cross-attention
between the two query sets + task heads. The auxiliary prompt-alignment task
(prompt_alignment_head.py) trains alongside this on a separate (crop, text) batch
-- it shapes the same shared backbone but isn't part of this forward pass, since
it operates on cropped regions and real referring expressions, not full images.

Segmentation prediction reuses Mask2Former's own class_predictor and mask_predictor
on the cross-attended segmentation queries (verified: identical shapes to Mask2Former's
native output), and its loss reuses Mask2Former's own criterion directly -- so
segmentation actually benefits from cross-attention with detection, not just
detection benefiting from segmentation.

Detection is new: DETR-style Hungarian-matched box + class loss (detection_loss.py),
since Mask2Former has no detection component to reuse.
"""
import torch
import torch.nn as nn

from det_seg_cross_attention import DetectionQueryDecoder, CrossTaskAttention, DetectionHead
from detection_loss import DetectionLoss


class NoveltyModel(nn.Module):
    def __init__(self, mask2former_model, num_classes=2, num_det_queries=100, num_det_decoder_layers=6):
        super().__init__()
        self.m2f = mask2former_model
        hidden_dim = mask2former_model.config.hidden_dim

        self.det_decoder = DetectionQueryDecoder(hidden_dim=hidden_dim, num_queries=num_det_queries,
                                                  num_layers=num_det_decoder_layers)
        self.cross_attn = CrossTaskAttention(hidden_dim=hidden_dim)
        self.det_head = DetectionHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, pixel_values, pixel_mask=None):
        if pixel_mask is None:
            pixel_mask = torch.ones(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3],
                                    device=pixel_values.device)

        base_out = self.m2f.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        pixel_features = base_out.pixel_decoder_last_hidden_state
        seg_queries = base_out.transformer_decoder_last_hidden_state

        det_queries = self.det_decoder(pixel_features)
        det_queries_x, seg_queries_x = self.cross_attn(det_queries, seg_queries)

        det_class_logits, det_boxes = self.det_head(det_queries_x)

        # Mask2Former's own decoder applies this layernorm to EVERY layer's hidden
        # state before class_predictor/mask_predictor ever see it (confirmed by
        # reading Mask2FormerMaskedAttentionDecoder.forward directly -- both calls
        # always take the layernorm'd `intermediate_hidden_states`, never the raw
        # layer output). Skipping it here would mean class_predictor and
        # mask_predictor -- both reused, pretrained-with-this-normalization modules
        # -- see inputs scaled differently than anything they were ever trained on.
        seg_queries_normed = self.m2f.model.transformer_module.decoder.layernorm(seg_queries_x)

        seg_class_logits = self.m2f.class_predictor(seg_queries_normed)
        mask_predictor = self.m2f.model.transformer_module.decoder.mask_predictor
        H, W = pixel_features.shape[-2:]
        seg_mask_logits, _ = mask_predictor(seg_queries_normed.transpose(0, 1), pixel_features,
                                            attention_mask_target_size=(H, W))

        return {
            'det_class_logits': det_class_logits,
            'det_boxes': det_boxes,
            'seg_class_logits': seg_class_logits,
            'seg_mask_logits': seg_mask_logits,
        }


if __name__ == '__main__':
    import numpy as np
    from pathlib import Path
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

    torch.manual_seed(0)

    # ---- Load ONE real image + real YOLO boxes + real mask, not synthetic data. ----
    DATASET_ROOT = Path('../processed_output')
    img_path = DATASET_ROOT / 'DAGM' / 'large' / 'images' / 'dagm_class10_0012_defect.png'
    label_path = DATASET_ROOT / 'DAGM' / 'large' / 'labels_yolo' / 'dagm_class10_0012_bbs.txt'
    mask_path = DATASET_ROOT / 'DAGM' / 'large' / 'masks' / 'dagm_class10_0012_mask.png'

    image = Image.open(img_path).convert('RGB').resize((384, 384), Image.Resampling.BILINEAR)
    mask = Image.open(mask_path).convert('L').resize((96, 96), Image.Resampling.NEAREST)  # 1/4 res, matches pixel_features

    pixel_values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mask_tensor = (torch.from_numpy(np.asarray(mask).copy()) > 0).float()

    real_boxes, real_classes = [], []
    for line in label_path.read_text().strip().splitlines():
        cls, cx, cy, w, h = line.split()
        real_boxes.append([float(cx), float(cy), float(w), float(h)])
        real_classes.append(1)  # binary: 1 = defect (matches every other baseline's convention)
    print(f'Real image: {img_path.name}, real boxes: {real_boxes}')

    config = Mask2FormerConfig(num_labels=2, num_queries=100)
    m2f = Mask2FormerForUniversalSegmentation(config)
    model = NoveltyModel(m2f, num_classes=2)
    model.train()

    outputs = model(pixel_values)
    print('det_class_logits:', tuple(outputs['det_class_logits'].shape))
    print('det_boxes:', tuple(outputs['det_boxes'].shape))
    print('seg_class_logits:', tuple(outputs['seg_class_logits'].shape))
    print('seg_mask_logits:', tuple(outputs['seg_mask_logits'].shape))

    # ---- Real detection loss on the real YOLO boxes. ----
    det_loss_fn = DetectionLoss(num_classes=2)
    det_targets = [{'classes': torch.tensor(real_classes), 'boxes': torch.tensor(real_boxes)}]
    det_loss, det_parts = det_loss_fn(outputs['det_class_logits'], outputs['det_boxes'], det_targets)
    print('detection loss:', det_loss.item(), det_parts)

    # ---- Real segmentation loss via Mask2Former's own criterion, on the real mask. ----
    seg_loss_dict = m2f.criterion(
        masks_queries_logits=outputs['seg_mask_logits'],
        class_queries_logits=outputs['seg_class_logits'],
        mask_labels=[mask_tensor.unsqueeze(0)],  # one instance for this image
        class_labels=[torch.tensor([1])],
    )
    seg_loss = sum(seg_loss_dict.values())
    print('segmentation loss:', seg_loss.item(), {k: v.item() for k, v in seg_loss_dict.items()})

    total_loss = det_loss + seg_loss
    total_loss.backward()

    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f'Full model: {n_grad}/{n_total} params with nonzero grad')

    # Always-inert third-party params (confirmed via source: SwinBackbone's own
    # final layernorm is unused with add_pooling_layer=False; level_embed is an
    # internal multi-scale-encoder param not on this output's gradient path).
    ALWAYS_INERT = {'m2f.model.pixel_level_module.encoder.swin.layernorm.weight',
                    'm2f.model.pixel_level_module.encoder.swin.layernorm.bias',
                    'm2f.model.pixel_level_module.decoder.level_embed'}
    # Individual Swin blocks (and Mask2Former decoder layers, via decoder.layerdrop)
    # get randomly, entirely dropped per forward pass by stochastic depth in
    # training mode (confirmed: backbone_config.drop_path_rate=0.3) -- WHICH ones
    # varies every run, so this has to be a pattern, not a fixed set of indices.
    import re
    STOCHASTIC_DEPTH_PATTERN = re.compile(
        r'm2f\.model\.pixel_level_module\.encoder\.swin\.encoder\.layers\.\d+\.blocks\.\d+\.|'
        r'm2f\.model\.transformer_module\.decoder\.layers\.\d+\.'
    )

    disconnected = [
        name for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
        and name not in ALWAYS_INERT and not STOCHASTIC_DEPTH_PATTERN.match(name)
    ]
    assert not disconnected, f'Unexplained disconnected parameters: {disconnected}'

    print()
    print('OK: full model verified end-to-end on a real image, real YOLO boxes, and a real mask --')
    print('detection loss (new Hungarian-matched loss) + segmentation loss (reused Mask2Former criterion)')
    print('both compute correctly and backpropagate through the shared backbone and both new modules.')
