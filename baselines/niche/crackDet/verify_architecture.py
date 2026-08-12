"""Local sanity-check script for the CrackDet reimplementation.

IMPORTANT -- has NOT been executed as part of building this package: this
development environment has no Python/PyTorch/e2cnn runtime available (no
local `python`, only the Windows Store app-execution-alias stub), and this
project's actual training/verification always happens on Kaggle or RunPod,
never locally (see every other *_kaggle.ipynb / *_runpod.py in this repo).
Every shape in this file was instead derived by hand, stage by stage,
against the ResNet-50 stage widths/strides and the e2cnn API as documented
in e2cnn's own docs/examples -- see the comments in model/backbone.py and
model/heads.py for the full derivation. Run this script yourself (`pip
install -r requirements.txt && python verify_architecture.py`) before
trusting it in a real training run -- treat it as unverified until it
actually prints PASSED on a real machine.

What it checks:
  1. piecewise_angle forward/inverse round-trip on representative angles
     in every branch (pure math, no torch needed for the assertions
     themselves beyond tensor plumbing).
  2. CrackDet(...) forward pass on a random batch produces every output
     tensor at the documented shape.
  3. A full forward + CrackDetLoss + backward pass reaches every trainable
     parameter with a non-zero, finite gradient (catches disconnected
     branches / accidental detach()'d ops).
  4. model/postprocess.decode(...) runs end-to-end on the model's own raw
     output without shape errors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from data.target_generator import CrackDetTargetGenerator, OrientedBox, collate_targets
from model.crackdet import CrackDet
from model.losses import CrackDetLoss
from model.piecewise_angle import decode_box, forward_transform
from model.postprocess import decode as decode_predictions


def check_piecewise_angle_roundtrip():
    print("[1/4] piecewise_angle forward/inverse round-trip ...")
    # One representative angle comfortably inside each of the 4 branches.
    thetas = torch.tensor([10.0, 30.0, 60.0, 100.0, 130.0, 170.0])
    h = torch.full_like(thetas, 40.0)
    w = torch.full_like(thetas, 12.0)

    theta_i, h_i, w_i, branch_idx = forward_transform(thetas, h, w)
    h_rt, w_rt, theta_rt = decode_box(h_i, w_i, theta_i, branch_idx)

    assert torch.allclose(theta_rt, thetas, atol=1e-3), (theta_rt, thetas)
    assert torch.allclose(h_rt, h, atol=1e-3), (h_rt, h)
    assert torch.allclose(w_rt, w, atol=1e-3), (w_rt, w)
    print("    PASSED -- theta/h/w recovered exactly for angles in every branch.")


def check_forward_shapes(model, batch=2, size=512):
    print("[2/4] forward-pass output shapes ...")
    x = torch.randn(batch, 3, size, size)
    out = model(x)
    fs = size // model.output_stride
    expected = {
        "heatmap": (batch, model.num_classes, fs, fs),
        "offset": (batch, 2, fs, fs),
        "size": (batch, model.num_branches, 2, fs, fs),
        "angle": (batch, model.num_branches, 1, fs, fs),
        "std": (batch, model.num_branches, 1, fs, fs),
    }
    for key, shape in expected.items():
        assert tuple(out[key].shape) == shape, f"{key}: got {tuple(out[key].shape)}, expected {shape}"
        assert torch.isfinite(out[key]).all(), f"{key} has non-finite values"
    assert (out["std"] > 0).all(), "std head must be strictly positive"
    print(f"    PASSED -- all 5 output tensors match the documented shape at stride {model.output_stride}.")
    return out


def check_gradients(model, size=512):
    print("[3/4] full forward + loss + backward, gradient coverage ...")
    x = torch.randn(2, 3, size, size, requires_grad=False)

    gen = CrackDetTargetGenerator(stride=model.output_stride, num_classes=model.num_classes)
    boxes_img0 = [OrientedBox(cx=100, cy=120, h=48, w=14, theta_deg=22.0),
                  OrientedBox(cx=300, cy=200, h=60, w=10, theta_deg=97.0)]
    boxes_img1 = [OrientedBox(cx=200, cy=256, h=30, w=9, theta_deg=161.0)]
    targets = collate_targets([gen((size, size), boxes_img0), gen((size, size), boxes_img1)])

    criterion = CrackDetLoss()
    preds = model(x)
    loss_dict = criterion(preds, targets)
    loss_dict["loss"].backward()

    n_params = 0
    n_missing_grad = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_params += 1
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0:
            n_missing_grad += 1
            print(f"    WARNING: no/zero/non-finite gradient on {name}")

    assert n_missing_grad == 0, f"{n_missing_grad}/{n_params} parameters got no usable gradient"
    print(f"    PASSED -- {n_params} parameter tensors all received a finite, non-zero gradient. "
          f"loss={float(loss_dict['loss']):.4f}")


def check_postprocess(model, size=512):
    print("[4/4] postprocess.decode(...) end-to-end ...")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, size, size)
        preds = model(x)
        decoded = decode_predictions(preds, topk=50, score_thresh=0.0, stride=model.output_stride)
    assert len(decoded) == 1
    assert decoded[0]["boxes"].shape[1] == 5
    print(f"    PASSED -- decoded {decoded[0]['boxes'].shape[0]} boxes with shape (N, 5).")


if __name__ == "__main__":
    check_piecewise_angle_roundtrip()
    torch.manual_seed(0)
    model = CrackDet(num_classes=1, group_order=8)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CrackDet instantiated: {n_params:,} parameters.")
    check_forward_shapes(model)
    check_gradients(model)
    check_postprocess(model)
    print("\nALL CHECKS PASSED.")
