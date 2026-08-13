# CrackDet (niche, faithful reimplementation)

Reimplementation of **"The Devil is in the Crack Orientation: A New
Perspective for Crack Detection"** (Chen, Zhang, Lai, Zhu, Liu, Chen, Li —
Shenzhen University, **ICCV 2023**). The paper's PDF is at
[`The Devil is in the Crack Orientation.pdf`](../../../The%20Devil%20is%20in%20the%20Crack%20Orientation.pdf)
in the repo root.

**This is not one of the project's 8 detection / 8 segmentation baselines.**
It lives under `baselines/niche/` on purpose — crack-specific methods
(DeepCrack, Crackformer) were explicitly cut from that list earlier in this
project, and CrackDet does a structurally different task (oriented
sub-crack detection, not axis-aligned bbox detection or pixel segmentation),
so it doesn't belong in either table. Built because it's a genuinely
interesting architecture worth having faithfully implemented, not to be
folded into the baseline comparison.

## There is no official code

Checked directly: no GitHub/GitLab repo under the authors' names, no link
from the CVF/IEEE/arXiv listings, and the paper's own text never mentions
a code release. Everything in `model/` is a from-scratch reimplementation
built strictly from the paper's equations, figures, and prose — not a
port of any existing repository (unlike, say, this project's PIDNet/GCNet
baselines, which vendor real official source).

## What's faithfully implemented

| Paper section | This package |
|---|---|
| Sec. 3.1, piecewise angle definition, Eq. 1, Fig. 2b, Eq. 6-7 | [`model/piecewise_angle.py`](model/piecewise_angle.py) — forward reprojection into 4 branches + the exact inverse maps Γ/Δ used at inference. Round-trip correctness (forward then inverse recovers the original θ, h, w exactly) was **hand-derived algebraically for all 4 branches** before writing any code (see the module docstring), and is asserted numerically in `verify_architecture.py`. |
| Sec. 3.2, ReEDNet backbone | [`model/backbone.py`](model/backbone.py) — e2cnn rotation-equivariant encoder-decoder, explicitly ResNet-50-shaped (stage depths `[3,4,6,3]`, bottleneck expansion 4, stage widths 64/128/256/512), matching Table 4's "ReED-R-50" label. Decoder is the standard 3-stage CenterNet deconv head (256→128→64), stride 32→4. |
| Sec. 3.2, multi-branch heads, Fig. 2c | [`model/heads.py`](model/heads.py) + [`model/crackdet.py`](model/crackdet.py) — shared heatmap/offset heads + 4 parallel branches, each with its own size/angle/angle-std heads. |
| Sec. 3.3, MAR Loss, Eq. 3-4 | [`model/losses.py`](model/losses.py) — the Wasserstein-distance term on the valid branch plus the variance-maximization term on the other 3 branches, in one `MARLoss` module. |
| Eq. 5, total objective | `CrackDetLoss`, same λ_off=0.1 / λ_size=0.2 / λ_MAR=0.1 as the paper (Sec. 4.1). |
| Sec. 3.4, variance-voting inference, Fig. 4 | [`model/postprocess.py`](model/postprocess.py) — CenterNet-style heatmap peak extraction, then per-peak `argmin_i σ_i²` branch selection, then the Γ/Δ inverse maps. |
| Sec. 4.1, training recipe | `train_crackdet_runpod.py` — Adam, lr 4e-4 decayed ×0.1 at epochs 20/40, 60 epochs, batch 32, 512×512 input (the ONPP/ORC/OCCSD recipe). |
| Sec. 4.1, evaluation metrics | Precision / Recall (rotated-box IoU ≥ 0.5 greedy matching) / MOE (mean orientation error, radians, computed on matched boxes only, accounting for a crack's 180°-not-360° angular symmetry) — matching Table 1/2's metric columns, not the project's usual detection/segmentation `summary.csv` schema (see below). |

## Where the paper is silent — documented design choices

The official implementation was never released, and the paper describes
the architecture at a level that leaves real implementation details
unspecified. Every such gap is called out inline (search each file for
"design choice" / "documented choice"), summarized here:

- **Rotation group order.** The paper never states the group order used
  by its e2cnn backbone. This package uses `N=8` (cyclic group C₈),
  matching ReDet's ReResNet (Han et al., CVPR'21) — the paper's own most
  directly comparable e2cnn-based equivariant backbone (it's cited and
  compared against throughout, e.g. Table 2), and the closest public
  reference point for what "e2cnn regular-representation ResNet-50"
  means in practice.
- **Head design (equivariant vs. invariant features).** The paper states
  the backbone should be rotation-equivariant so *both* size and angle
  regression benefit (Sec. 3.2), but also says each head is "simply...
  a fully-connected layer" (not itself an equivariant module). This
  package resolves that by giving each of the 4 branches its own small
  equivariant conv trunk (so branch-specific features stay
  rotation-equivariant, matching the stated rationale), then
  group-pools to an invariant per-pixel vector before the literal
  FC/1×1-conv regression heads. A fully group-representation-correct
  equivariant *angle* head (where rotating the input by g provably
  shifts the predicted angle by g, via an irrep rather than a pooled
  scalar) is a harder, still-open design problem this paper doesn't
  spell out either — not attempted here.
- **Exact channel widths / decoder skip connections.** Backbone widths
  are pinned to real ResNet-50 (justified by the "ReED-R-50" label in
  Table 4); the decoder has no U-Net-style skip fusion, matching
  vanilla CenterNet's plain deconv stack (Fig. 2c shows no skip
  arrows).
- **Gaussian heatmap radius for oriented boxes.** Uses the standard
  CornerNet/CenterNet radius formula (Law & Deng, ECCV'18) applied to a
  box's own (h, w) — the paper explicitly builds on CenterNet for this
  part and doesn't propose an orientation-aware variant.

## No dataset

ONPP, ORC, and OCCSD (the paper's own datasets, Sec. 3.5) were never
released publicly, and this project currently has no oriented sub-crack
box annotations of its own (existing data here is pixel-mask / axis-
aligned-box only — see `dataset_explorer/`). [`data/dataset.py`](data/dataset.py)
therefore defines an explicit JSON schema (documented in its module
docstring) rather than hardcoding a source; `train_crackdet_runpod.py`
requires `TRAIN_ANNOTATIONS` / `VAL_ANNOTATIONS` / `TEST_ANNOTATIONS` env
vars pointing at files in that schema and exits with an explanatory error
if they're not set. **This code has not been trained or run on real data.**

[`data/split_dataset.py`](data/split_dataset.py) does the 8:1:1 split
(Sec. 4.1) — plain random shuffle at that ratio, seeded. The paper doesn't
describe the splitting procedure beyond the ratio itself (no stratification
variable, no seed given), so that's exactly what this does and nothing
more; flagged in its own docstring that a plain random split can land
imbalanced by chance if box density varies a lot patch-to-patch, since the
paper gives no stratification rule to prevent that.

**Preprocessing matches the paper exactly, not this project's other
baseline scripts.** SegNeXt/PIDNet/GCNet resize raw images to a fixed size
at training time via an independent (sx, sy) stretch — fine for a pixel
mask, but it would silently corrupt an oriented box's angle (a non-uniform
stretch turns a rotated rectangle into a non-rectangular parallelogram).
The paper never resizes at all: Sec. 3.5 slices full-resolution source
images into non-overlapping 512×512 patches *once*, as a dataset-
construction step, and trains on those patches directly. `dataset.py`
mirrors that — it expects every image to already be exactly `input_size` ×
`input_size` and raises a hard error otherwise, rather than silently
resizing/padding it into shape. [`data/slice_dataset.py`](data/slice_dataset.py)
replicates that dataset-construction step (full-resolution images + box
annotations → fixed-size patches, boxes translated to patch-local
coordinates, angle/side-lengths untouched since a crop is a pure
translation). One thing it can't reproduce exactly: the paper's reported
sample counts (e.g. ONPP: 3,104 patches from 200 images) are far smaller
than exhaustive non-overlapping tiling of their stated source resolutions
would produce, meaning their pipeline drops most background-only tiles
somewhere without spelling out the exact rule — `slice_dataset.py` defaults
to dropping empty (0-box) patches as the closest match to that, flagged
explicitly in its own docstring as an inferred, not verbatim, choice.

## Verification status — be precise about what "faithful" means here

This development environment has **no local Python/PyTorch/e2cnn runtime**
(confirmed: no working `python`/`pip`, only a Windows Store alias stub —
every other script in this repo runs on Kaggle or RunPod, never locally,
and that held here too). So:

- The **math** (`piecewise_angle.py`'s forward/inverse transforms) was
  verified by hand, algebraically, for a representative angle in each of
  the 4 branches — reproduced in `verify_architecture.py`'s first check.
- Every **tensor shape** through the backbone/heads/losses/postprocess
  was derived by hand, stage by stage (stem → 4 encoder stages → 3
  decoder stages → heads), cross-checked against real ResNet-50
  stride/channel arithmetic. These are documented as comments at each
  shape-changing point in `backbone.py`/`heads.py`.
- **None of this has actually been executed.** `verify_architecture.py`
  is written and ready (shape assertions, a full forward+loss+backward
  pass checking every parameter gets a gradient, and an end-to-end
  `postprocess.decode()` smoke test) but has not been run, because there's
  no runtime available in this session to run it on. Treat the
  architecture as **unverified until someone runs `verify_architecture.py`
  on a real machine** (`pip install -r requirements.txt && python
  verify_architecture.py`) — most likely the first thing to do on
  whatever RunPod/Kaggle box eventually trains this.

## Layout

```
baselines/niche/crackDet/
  model/
    piecewise_angle.py   piecewise angle definition, forward + inverse (Γ, Δ)
    backbone.py           ReEDNet: e2cnn rotation-equivariant ResNet-50 + CenterNet decoder
    heads.py               heatmap/offset heads + 4 piecewise-angle branch heads
    crackdet.py             full model, backbone + heads
    losses.py               L_k (focal), L_off, L_size, MAR loss, Eq. 5 total
    postprocess.py         heatmap peak extraction + variance-voting decode
  data/
    target_generator.py   oriented boxes -> Gaussian heatmap + per-branch regression targets
    dataset.py               OrientedCrackDataset (documents the required JSON schema)
    slice_dataset.py        replicates the paper's Sec. 3.5 dataset construction (full-res images -> fixed-size patches)
    split_dataset.py        paper's 8:1:1 train/val/test split (Sec. 4.1), seeded random shuffle
  train_crackdet_runpod.py  training script (Adam, paper's LR schedule, W&B logging)
  verify_architecture.py    shape/gradient/round-trip sanity checks (NOT yet run, see above)
  requirements.txt
```
