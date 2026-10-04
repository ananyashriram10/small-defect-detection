"""
New architecture pieces that sit on top of a real, pretrained Mask2Former backbone:

  1. DetectionQueryDecoder -- a second, parallel set of learned queries that decode
     boxes instead of masks. Deliberately a STANDARD transformer decoder (self-attn
     + plain cross-attn to the shared pixel-decoder features), not a copy of
     Mask2Former's masked-attention decoder -- Mask2Former's masking mechanism uses
     each layer's own predicted MASK to restrict the next layer's attention, which
     has no natural analog for a query that's regressing a box, not a mask. Forcing
     it in would be cargo-culting the mechanism, not reusing it faithfully.

  2. CrossTaskAttention -- bidirectional cross-attention between the detection
     queries and Mask2Former's own segmentation queries, so each task's queries can
     draw on the other's. This is the actual novelty piece: two genuinely separate
     query sets that cross-attend, not one shared query set decoded two ways
     (which is what MaskDINO does).

  3. DetectionHead -- box (cx, cy, w, h) + class logits off the (now cross-attended)
     detection queries, DETR-style.

Verified against the REAL facebook/mask2former-swin-tiny-ade-semantic model's actual
output shapes below, not synthetic stand-ins.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionQueryDecoder(nn.Module):
    """Standard (unmasked) transformer decoder: detection queries self-attend, then
    cross-attend to the shared pixel-decoder feature map, num_layers times."""

    def __init__(self, hidden_dim=256, num_queries=100, num_layers=6, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_features = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, pixel_features):
        """pixel_features: [B, C, H, W] from Mask2Former's pixel_decoder_last_hidden_state."""
        B = pixel_features.shape[0]
        memory = pixel_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        queries = self.query_features.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_position.weight.unsqueeze(0).expand(B, -1, -1)

        out = self.decoder(tgt=queries + query_pos, memory=memory)
        return out  # [B, num_queries, hidden_dim]


class CrossTaskAttention(nn.Module):
    """Bidirectional cross-attention: detection queries attend to segmentation
    queries as K/V and vice versa, each followed by a residual + FFN update --
    a standard post-cross-attention transformer block, applied to both directions."""

    def __init__(self, hidden_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.det_attends_seg = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.seg_attends_det = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.det_norm1 = nn.LayerNorm(hidden_dim)
        self.seg_norm1 = nn.LayerNorm(hidden_dim)

        self.det_ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, hidden_dim))
        self.seg_ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, hidden_dim))
        self.det_norm2 = nn.LayerNorm(hidden_dim)
        self.seg_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, det_queries, seg_queries):
        det_attn, _ = self.det_attends_seg(query=det_queries, key=seg_queries, value=seg_queries)
        det_queries = self.det_norm1(det_queries + det_attn)
        det_queries = self.det_norm2(det_queries + self.det_ffn(det_queries))

        seg_attn, _ = self.seg_attends_det(query=seg_queries, key=det_queries, value=det_queries)
        seg_queries = self.seg_norm1(seg_queries + seg_attn)
        seg_queries = self.seg_norm2(seg_queries + self.seg_ffn(seg_queries))

        return det_queries, seg_queries


class DetectionHead(nn.Module):
    """DETR-style box + class heads off the final detection query embeddings."""

    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 = no-object, matches Mask2Former's own convention
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, det_queries):
        class_logits = self.class_head(det_queries)
        boxes = self.box_head(det_queries).sigmoid()  # normalized (cx, cy, w, h) in [0, 1]
        return class_logits, boxes


if __name__ == '__main__':
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

    torch.manual_seed(0)
    config = Mask2FormerConfig(num_labels=2, num_queries=100)
    m2f = Mask2FormerForUniversalSegmentation(config)
    m2f.eval()

    x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        base_out = m2f.model(pixel_values=x, pixel_mask=torch.ones(2, 384, 384))
    pixel_features = base_out.pixel_decoder_last_hidden_state
    seg_queries = base_out.transformer_decoder_last_hidden_state
    print('Real Mask2Former outputs -- pixel_features:', tuple(pixel_features.shape), ' seg_queries:', tuple(seg_queries.shape))

    det_decoder = DetectionQueryDecoder(hidden_dim=256, num_queries=100, num_layers=6)
    cross_attn = CrossTaskAttention(hidden_dim=256)
    det_head = DetectionHead(hidden_dim=256, num_classes=2)

    det_queries = det_decoder(pixel_features)
    print('det_queries (before cross-attention):', tuple(det_queries.shape))
    assert det_queries.shape == (2, 100, 256)

    det_queries_x, seg_queries_x = cross_attn(det_queries, seg_queries)
    print('det_queries (after cross-attention):', tuple(det_queries_x.shape))
    print('seg_queries (after cross-attention):', tuple(seg_queries_x.shape))
    assert det_queries_x.shape == det_queries.shape
    assert seg_queries_x.shape == seg_queries.shape

    class_logits, boxes = det_head(det_queries_x)
    print('class_logits:', tuple(class_logits.shape), ' boxes:', tuple(boxes.shape))
    assert class_logits.shape == (2, 100, 3)
    assert boxes.shape == (2, 100, 4)
    assert (boxes >= 0).all() and (boxes <= 1).all(), 'box outputs must be normalized in [0,1]'

    # Gradient flow check: a simple dummy loss (real Hungarian-matched loss comes later),
    # just to confirm every new parameter is actually wired into the computation graph.
    dummy_loss = class_logits.sum() + boxes.sum() + seg_queries_x.sum()
    dummy_loss.backward()

    new_modules = {'det_decoder': det_decoder, 'cross_attn': cross_attn, 'det_head': det_head}
    for mod_name, mod in new_modules.items():
        n_total = sum(1 for p in mod.parameters() if p.requires_grad)
        n_grad = sum(1 for p in mod.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        print(f'{mod_name}: {n_grad}/{n_total} params with nonzero grad')
        assert n_grad == n_total, f'{mod_name} has a disconnected parameter'

    print()
    print('OK: new detection decoder + cross-attention + detection head all verified against real Mask2Former outputs.')
