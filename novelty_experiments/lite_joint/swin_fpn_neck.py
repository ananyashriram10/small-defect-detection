"""P2-P5 FPN+PAN neck feeding the lightweight detection head directly from Mask2Former's Swin
backbone -- the detection-side counterpart to the existing (unmodified) pixel decoder segmentation
already uses. Confirmed (2026-08-21, against the real facebook/mask2former-swin-tiny-ade-semantic
checkpoint) that model.model.pixel_level_module(pixel_values, output_hidden_states=True) returns
encoder_hidden_states as a 4-tuple of the RAW per-Swin-stage feature maps -- (1,96,160,160),
(1,192,80,80), (1,384,40,40), (1,768,20,20), i.e. strides 4/8/16/32 -- from the exact same forward
pass that already produces decoder_last_hidden_state for segmentation. So detection taps this
tuple directly; it costs one extra neck+head forward, not a second backbone pass.

C2fBlock/ConvBNAct structurally mirror YOLOv8's well-documented, public neck design (same
family yolov8n_p2pan.yaml already uses in this project's own strongest detection baseline) --
reimplemented from scratch here the same way detection_loss.py already reimplements DETR's
Hungarian loss from scratch, not imported from ultralytics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ch, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNAct(ch, ch, k=3)
        self.cv2 = ConvBNAct(ch, ch, k=3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C2fBlock(nn.Module):
    """Split -> stack of bottlenecks -> concat all intermediate outputs -> fuse. The standard
    CSP-style block this whole neck family is built from."""
    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        self.hidden = out_ch // 2
        self.cv1 = ConvBNAct(in_ch, 2 * self.hidden, k=1)
        self.blocks = nn.ModuleList([Bottleneck(self.hidden, shortcut) for _ in range(n)])
        self.cv2 = ConvBNAct((2 + n) * self.hidden, out_ch, k=1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SwinPAFPNNeck(nn.Module):
    """Takes the 4 raw Swin stage outputs (channels 96/192/384/768, strides 4/8/16/32 for
    Swin-Tiny at any input size), projects each to a common lightweight width, fuses top-down
    then bottom-up, and returns 4 feature maps at the same 4 strides -- the P2-P5 pyramid the
    detection head needs."""
    def __init__(self, in_channels=(96, 192, 384, 768), neck_ch=128, n=1):
        super().__init__()
        self.neck_ch = neck_ch
        self.lateral = nn.ModuleList([ConvBNAct(c, neck_ch, k=1) for c in in_channels])

        # top-down: P5 -> P4 -> P3 -> P2, each step upsamples + concats with the lateral + fuses
        self.td_fuse = nn.ModuleList([C2fBlock(2 * neck_ch, neck_ch, n=n, shortcut=False) for _ in range(3)])

        # bottom-up: P2 -> P3 -> P4 -> P5, each step downsamples + concats with the td feature + fuses
        self.down = nn.ModuleList([ConvBNAct(neck_ch, neck_ch, k=3, s=2) for _ in range(3)])
        self.bu_fuse = nn.ModuleList([C2fBlock(2 * neck_ch, neck_ch, n=n, shortcut=False) for _ in range(3)])

    def forward(self, stage_features):
        """stage_features: [P2_raw, P3_raw, P4_raw, P5_raw] (strides 4/8/16/32). Returns
        [P2_out, P3_out, P4_out, P5_out] at the same 4 strides, each with neck_ch channels."""
        p2, p3, p4, p5 = [lat(f) for lat, f in zip(self.lateral, stage_features)]

        p5_td = p5
        p4_td = self.td_fuse[0](torch.cat([F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest"), p4], dim=1))
        p3_td = self.td_fuse[1](torch.cat([F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest"), p3], dim=1))
        p2_out = self.td_fuse[2](torch.cat([F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest"), p2], dim=1))

        p3_out = self.bu_fuse[0](torch.cat([self.down[0](p2_out), p3_td], dim=1))
        p4_out = self.bu_fuse[1](torch.cat([self.down[1](p3_out), p4_td], dim=1))
        p5_out = self.bu_fuse[2](torch.cat([self.down[2](p4_out), p5_td], dim=1))

        return [p2_out, p3_out, p4_out, p5_out]


if __name__ == "__main__":
    from transformers import Mask2FormerForUniversalSegmentation

    torch.manual_seed(0)
    MODEL_NAME = "facebook/mask2former-swin-tiny-ade-semantic"

    print(f"Loading real checkpoint {MODEL_NAME} and running one real forward pass...")
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
    m2f.train()

    pixel_values = torch.randn(1, 3, 640, 640, requires_grad=False)
    base_out = m2f.model.pixel_level_module(pixel_values, output_hidden_states=True)

    stage_features = list(base_out.encoder_hidden_states)
    expected_shapes = [(1, 96, 160, 160), (1, 192, 80, 80), (1, 384, 40, 40), (1, 768, 20, 20)]
    for feat, exp in zip(stage_features, expected_shapes):
        assert tuple(feat.shape) == exp, f"raw stage shape changed vs. the confirmed baseline: got {tuple(feat.shape)}, expected {exp}"
    print("Raw Swin stage shapes match the confirmed baseline exactly:", [tuple(f.shape) for f in stage_features])

    print("\nAlso confirming segmentation's own path is untouched by this (same forward pass, different fields):")
    print("decoder_last_hidden_state:", tuple(base_out.decoder_last_hidden_state.shape))

    print("\nChecking whether the TOP-LEVEL m2f.model(...) call (the one novelty_model.py's existing")
    print("forward() already uses, which does accept pixel_mask) also exposes encoder_hidden_states")
    print("-- if so, LiteJointModel can reuse that exact call pattern with output_hidden_states=True")
    print("added, rather than bypassing it to call pixel_level_module directly and losing pixel_mask.")
    pixel_mask = torch.ones(1, 640, 640)
    top_out = m2f.model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_hidden_states=True)
    has_encoder_hidden_states = getattr(top_out, "encoder_hidden_states", None) is not None
    print("Top-level output has encoder_hidden_states:", has_encoder_hidden_states)
    if has_encoder_hidden_states:
        print("  shapes:", [tuple(f.shape) for f in top_out.encoder_hidden_states])
        assert [tuple(f.shape) for f in top_out.encoder_hidden_states] == expected_shapes, \
            "top-level call's encoder_hidden_states should match pixel_level_module's directly"
        print("  Confirmed identical to the direct pixel_level_module call -- LiteJointModel should use")
        print("  m2f.model(pixel_values, pixel_mask, output_hidden_states=True) as its one shared call.")
    else:
        print("  Not exposed at the top level -- LiteJointModel will need to call pixel_level_module")
        print("  directly (without pixel_mask) instead, same as this self-test does above.")

    neck = SwinPAFPNNeck(in_channels=(96, 192, 384, 768), neck_ch=128, n=1)
    neck.train()
    outputs = neck(stage_features)
    expected_out_shapes = [(1, 128, 160, 160), (1, 128, 80, 80), (1, 128, 40, 40), (1, 128, 20, 20)]
    for out, exp in zip(outputs, expected_out_shapes):
        assert tuple(out.shape) == exp, f"neck output shape wrong: got {tuple(out.shape)}, expected {exp}"
    print("\nNeck output shapes (P2-P5):", [tuple(o.shape) for o in outputs])

    print("\nGradient coverage check: backward through a dummy loss on all 4 outputs, into both the neck AND the Swin backbone")
    loss = sum(o.sum() for o in outputs)
    loss.backward()

    neck_total = sum(1 for p in neck.parameters() if p.requires_grad)
    neck_grad = sum(1 for p in neck.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Neck: {neck_grad}/{neck_total} params with nonzero grad")
    assert neck_grad == neck_total, "every neck parameter must receive gradient from all 4 output levels combined"

    backbone_params = [p for n, p in m2f.named_parameters() if "pixel_level_module.encoder" in n and p.requires_grad]
    backbone_grad = sum(1 for p in backbone_params if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Swin backbone: {backbone_grad}/{len(backbone_params)} params with nonzero grad "
          f"(some will legitimately be 0 -- stochastic depth drops random blocks per forward pass, "
          f"same caveat already documented in novelty_model.py's own self-test)")
    assert backbone_grad > 0.5 * len(backbone_params), "the backbone should receive gradient through most of its parameters via the raw stage taps"

    print("\nOK: SwinPAFPNNeck verified against the real checkpoint -- correct shapes at all 4 levels,")
    print("full gradient coverage through the neck, and real gradient flowing back into the shared Swin backbone.")
