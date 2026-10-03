"""
SegNeXt-T (MSCAN-T backbone), binary defect segmentation -- RunPod version.

Faithful port of baselines/segmentation/SegNeXt-T/segnext_t_kaggle.ipynb: same
vendored MSCAN + LightHamHead architecture (no MMSegmentation dependency --
see that notebook's markdown cell for why), same dataset split, same
pretrained-backbone download-with-fallback, same training loop, same
D-FINE-matching summary.csv schema. Only the paths changed (Kaggle ->
RunPod) and the dataset now comes from the same local-tar-or-HF-download
pattern as train_dfine_s_runpod.py.

Run:
    export WANDB_API_KEY=<your wandb key>       # optional, else interactive login prompt
    export HF_TOKEN=<your huggingface token>    # only needed if DATASET_ROOT isn't already populated
    export DATASET_ROOT=/workspace/dataset      # optional, this is the default
    nohup python -u train_segnext_t_runpod.py > train.log 2>&1 &
    tail -f train.log

Same params as the Kaggle notebook, unchanged:
    MAX_EPOCHS=50, PATIENCE=15 (bumped from the notebook's original 10 --
    proportionally more patient than RT-DETRv2's 15/100 given SegNeXt's
    smaller 50-epoch budget), batch 8, lr 6e-5, weight decay 1e-2, AMP on
    (no known instability here, unlike D-FINE's documented NaN-box issue).
"""

import os

# Defensive, same reasoning as every other RunPod script in this project --
# costs nothing if multi-GPU auto-wrap wasn't going to happen anyway, and it
# already bit one run here once (HF Trainer + DataParallel on RT-DETRv2).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# Config -- same values as segnext_t_kaggle.ipynb
# ============================================================
RUN_NAME = "RunL_segnext_t_imgsz640"
MODEL_LABEL = "SegNeXt-T"
WANDB_PROJECT = "smallDefectDetection"
WANDB_RUN_NAME = f"{MODEL_LABEL}_segmentation"

HF_DATASET_REPO = "Smalldefect/SmallDefectDataseet"  # private -- needs HF_TOKEN if downloading fresh

DATASET_NAMES = ["DAGM", "GC10-DET", "KolektorSDD2", "MPDD", "MTD", "Severstal", "VisA"]
SIZE_BUCKETS = ["small", "medium", "large"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMG_SIZE = 640
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE = 15
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 2

# Official MSCAN-T backbone, ImageNet-1K pretrained then ADE20K-adapted (160k iters).
# Only the backbone.* weights get used -- the ADE20K decode head (150 classes) is discarded.
PRETRAINED_URL = "https://cloud.tsinghua.edu.cn/f/5da98841b8384ba0988a/?dl=1"

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/workspace/dataset"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "/workspace/segnext_run"))
RUN_DIR = BASE_DIR / "segnext_runs" / RUN_NAME
FINAL_OUTPUT_DIR = BASE_DIR / "final_outputs" / RUN_NAME

random.seed(SEED)

# ============================================================
# numpy/pandas/Pillow/wandb are not guaranteed preinstalled on every RunPod
# base image (confirmed the hard way on the D-FINE run -- pandas was missing).
# torch is not defended against: if that's missing this template is unusable
# anyway, so it's not worth the extra check.
# ============================================================
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "numpy", "pandas", "pillow", "wandb"],
    check=True,
)

# ============================================================
# CUDA / numpy sanity check. Same class of bug already happened once on
# this project's Kaggle runs ("ms detr") -- catch it here, in seconds, not
# partway through a run.
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

print("torch:", torch.__version__, "| torch's CUDA build:", torch.version.cuda)
print("numpy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count(), "(should be 1)")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

_smoke = torch.from_numpy(np.zeros(3, dtype=np.float32))
if torch.cuda.is_available():
    _smoke = (_smoke.cuda() + 1).cpu()
print("numpy <-> torch <-> CUDA smoke test passed.")

if not torch.cuda.is_available():
    sys.exit("CUDA is not available on this pod. Check the GPU is attached and the driver/CUDA toolkit is loaded (`nvidia-smi`).")
if torch.cuda.device_count() != 1:
    sys.exit(f"Expected exactly 1 visible GPU, found {torch.cuda.device_count()} despite CUDA_VISIBLE_DEVICES=0 -- investigate before continuing.")

DEVICE = torch.device("cuda")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def wandb_login_anywhere():
    # Works on RunPod (env var) and Kaggle (Secrets add-on, silently skipped
    # here since the kaggle_secrets module won't exist on a pod) without ever
    # hardcoding the key.
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            api_key = None
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()


wandb_login_anywhere()
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "model": MODEL_LABEL,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    },
)

# ============================================================
# Dataset: same download-if-missing pattern as train_dfine_s_runpod.py --
# checks for images/masks/labels_yolo dirs, downloads from the private HF
# repo with retry-with-backoff if not already present. Skips entirely if
# you scp'd + untarred dataset.tar to DATASET_ROOT beforehand, same as every
# other run in this project.
# ============================================================
def missing_dataset_dirs():
    return [
        f"{name}/{size}"
        for name in DATASET_NAMES
        for size in SIZE_BUCKETS
        if not (DATASET_ROOT / name / size / "images").exists()
        or not (DATASET_ROOT / name / size / "masks").exists()
        or not (DATASET_ROOT / name / size / "labels_yolo").exists()
    ]


print(f"Checking dataset at {DATASET_ROOT} ...")
missing = missing_dataset_dirs()

if missing:
    print(f"Dataset incomplete/missing at {DATASET_ROOT} ({len(missing)} "
          f"bucket(s) missing) -- attempting download from {HF_DATASET_REPO}.")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "HF_TOKEN is not set and the dataset isn't already on this pod. "
            "The dataset repo is private, so downloading it requires a token:\n"
            "  export HF_TOKEN=<your huggingface token>\n"
            "then re-run. Alternatively, scp a pre-built dataset.tar to this pod "
            "and extract it to DATASET_ROOT yourself before running this script -- "
            "it'll skip the download entirely if the expected structure is already there."
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "--no-cache-dir", "huggingface_hub"],
        check=True,
    )
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
    from huggingface_hub import snapshot_download

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_RETRIES = 6
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir=str(DATASET_ROOT),
                token=hf_token,
                max_workers=4,
            )
            break
        except Exception as error:
            if attempt == DOWNLOAD_RETRIES:
                raise
            wait_seconds = 30 * attempt
            print(f"Download attempt {attempt}/{DOWNLOAD_RETRIES} failed ({error}); "
                  f"waiting {wait_seconds}s and retrying. snapshot_download resumes "
                  "from already-fetched files, it doesn't restart from zero.")
            time.sleep(wait_seconds)
    print("Download finished. Re-checking structure...")
    missing = missing_dataset_dirs()

if missing:
    sys.exit(
        f"Dataset still incomplete at {DATASET_ROOT}. Missing images/masks/labels_yolo under: "
        f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
    )
print("Dataset structure looks complete.")

# ============================================================
# Match images + masks + labels (identical logic to segnext_t_kaggle.ipynb)
# ============================================================
def index_files(directory, suffixes):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def candidate_stems(image_stem, target):
    base = image_stem.removesuffix("_defect")
    if target == "mask":
        return [image_stem, image_stem.replace("_defect", "_mask"), base, base + "_mask", base + "_gt"]
    return [image_stem, image_stem.replace("_defect", "_bbs"), base, base + "_bbs"]


samples = []
missing_count = 0
for dataset_name in DATASET_NAMES:
    for size_bucket in SIZE_BUCKETS:
        bucket_root = DATASET_ROOT / dataset_name / size_bucket
        image_dir = bucket_root / "images"
        mask_dir = bucket_root / "masks"
        label_dir = bucket_root / "labels_yolo"

        mask_index = index_files(mask_dir, IMAGE_EXTS)
        label_index = index_files(label_dir, {".txt"})
        matched = 0
        bucket_missing = 0

        for image_path in image_dir.iterdir():
            # Skip macOS AppleDouble metadata sidecars (._<name>.<ext>) -- same
            # bug this project already hit once, doubling the dataset with junk.
            if image_path.suffix.lower() not in IMAGE_EXTS or image_path.name.startswith("._"):
                continue
            mask_path = next((mask_index[s] for s in candidate_stems(image_path.stem, "mask") if s in mask_index), None)
            label_path = next((label_index[s] for s in candidate_stems(image_path.stem, "box") if s in label_index), None)
            if mask_path is None or label_path is None:
                bucket_missing += 1
                continue
            samples.append({
                "image_path": image_path, "mask_path": mask_path,
                "dataset": dataset_name, "size": size_bucket,
                "stratum": f"{dataset_name}_{size_bucket}",
            })
            matched += 1
        missing_count += bucket_missing
        print(f"{dataset_name}/{size_bucket}: {matched} matched, {bucket_missing} missing")

print("Total matched:", len(samples), "| missing:", missing_count)
if len(samples) != 12670:
    sys.exit(f"Expected 12,670 image-mask-label triplets, found {len(samples)}.")

# ============================================================
# Stratified split (identical seed/ratios to every other script here)
# ============================================================
by_stratum = defaultdict(list)
for sample in samples:
    by_stratum[sample["stratum"]].append(sample)

rng = random.Random(SEED)
train_samples, val_samples, test_samples = [], [], []
for _, group in sorted(by_stratum.items()):
    group = list(group)
    rng.shuffle(group)
    train_end = int(len(group) * TRAIN_RATIO)
    val_end = train_end + int(len(group) * VAL_RATIO)
    train_samples.extend(group[:train_end])
    val_samples.extend(group[train_end:val_end])
    test_samples.extend(group[val_end:])

rng.shuffle(train_samples)
rng.shuffle(val_samples)
rng.shuffle(test_samples)

test_sets = {
    "overall": test_samples,
    "small": [s for s in test_samples if s["size"] == "small"],
    "medium": [s for s in test_samples if s["size"] == "medium"],
    "large": [s for s in test_samples if s["size"] == "large"],
}


def print_counts(name, split):
    counts = Counter(s["size"] for s in split)
    print(f"{name}: total={len(split)} small={counts['small']} medium={counts['medium']} large={counts['large']}")


print_counts("Train", train_samples)
print_counts("Validation", val_samples)
for name, split in test_sets.items():
    print_counts("Test " + name, split)

assert len(train_samples) == 8858
assert len(val_samples) == 1892
assert len(test_samples) == 1920

# ============================================================
# Dataset / DataLoader
# ============================================================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_binary_mask(mask_path, target_size):
    mask = Image.open(mask_path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


class DefectMaskDataset(Dataset):
    def __init__(self, split_samples):
        self.samples = split_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        mask = load_binary_mask(sample["mask_path"], (IMG_SIZE, IMG_SIZE))
        pixel_values = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD
        labels = torch.from_numpy(mask).long()
        return {"pixel_values": pixel_values, "labels": labels}


def make_loader(split_samples, shuffle=False):
    return DataLoader(
        DefectMaskDataset(split_samples),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )


train_loader = make_loader(train_samples, shuffle=True)
val_loader = make_loader(val_samples)
print("Train batches:", len(train_loader), "Validation batches:", len(val_loader))

# ---- Vendored from the official SegNeXt repo (Visual-Attention-Network/SegNeXt), ----
# ---- NeurIPS 2022 Spotlight. mmcv/mmseg-specific building blocks (BaseModule, ----
# ---- build_norm_layer, ConvModule, the BACKBONES/HEADS registries) are swapped ----
# ---- for plain PyTorch equivalents; the forward-pass math is unchanged from the ----
# ---- paper's mscan.py / ham_head.py. ----

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_chans=64, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAN(nn.Module):
    """MSCAN-T backbone. Ported from Visual-Attention-Network/SegNeXt (NeurIPS 2022)."""

    def __init__(self, in_chans=3, embed_dims=(32, 64, 160, 256), mlp_ratios=(8, 8, 4, 4),
                 drop_rate=0., drop_path_rate=0.1, depths=(3, 3, 5, 2), num_stages=4):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_chans, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[i - 1], embed_dim=embed_dims[i])
            block = nn.ModuleList([
                Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, groups=32):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not norm)
        self.norm = nn.GroupNorm(groups, out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class NMF2D(nn.Module):
    """Non-negative Matrix Factorization ('Hamburger') module. MD_R=16 per the official tiny config."""

    def __init__(self, md_r=16, md_s=1, train_steps=6, eval_steps=7):
        super().__init__()
        self.S = md_s
        self.R = md_r
        self.train_steps = train_steps
        self.eval_steps = eval_steps

    def _build_bases(self, B, S, D, R, device):
        bases = torch.rand((B * S, D, R), device=device)
        return F.normalize(bases, dim=1)

    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        return coef * numerator / (denominator + 1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        D = C // self.S
        N = H * W
        x_flat = x.view(B * self.S, D, N)

        bases = self._build_bases(B, self.S, D, self.R, x.device)
        coef = torch.bmm(x_flat.transpose(1, 2), bases)
        coef = F.softmax(coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x_flat, bases, coef)
        coef = self.compute_coef(x_flat, bases, coef)

        out = torch.bmm(bases, coef.transpose(1, 2))
        return out.view(B, C, H, W)


class Hamburger(nn.Module):
    def __init__(self, ham_channels=256, md_r=16):
        super().__init__()
        self.ham_in = ConvModule(ham_channels, ham_channels, 1, norm=False, act=False)
        self.ham = NMF2D(md_r=md_r)
        self.ham_out = ConvModule(ham_channels, ham_channels, 1, norm=True, act=False)

    def forward(self, x):
        enjoy = F.relu(self.ham_in(x), inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        return F.relu(x + enjoy, inplace=True)


class LightHamHead(nn.Module):
    """LightHamHead decoder. Ported from Visual-Attention-Network/SegNeXt (NeurIPS 2022)."""

    def __init__(self, in_channels=(64, 160, 256), in_index=(1, 2, 3), channels=256,
                 ham_channels=256, num_classes=2, dropout_ratio=0.1, md_r=16):
        super().__init__()
        self.in_index = in_index
        self.align_corners = False
        self.squeeze = ConvModule(sum(in_channels), ham_channels, 1)
        self.hamburger = Hamburger(ham_channels, md_r=md_r)
        self.align = ConvModule(ham_channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        inputs = [inputs[i] for i in self.in_index]
        inputs = [F.interpolate(level, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
                  for level in inputs]
        x = self.squeeze(torch.cat(inputs, dim=1))
        x = self.hamburger(x)
        x = self.align(x)
        x = self.dropout(x)
        return self.conv_seg(x)


class SegNeXt(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = MSCAN(embed_dims=(32, 64, 160, 256), mlp_ratios=(8, 8, 4, 4),
                              depths=(3, 3, 5, 2), drop_path_rate=0.1)
        self.decode_head = LightHamHead(in_channels=(64, 160, 256), in_index=(1, 2, 3),
                                        channels=256, ham_channels=256,
                                        num_classes=num_classes, dropout_ratio=0.1, md_r=16)

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=pixel_values.shape[2:], mode='bilinear', align_corners=False)
        return logits


print('SegNeXt-T architecture defined (vendored, no MMSegmentation dependency).')

model = SegNeXt(num_classes=2).to(DEVICE)

pretrained_loaded = False
try:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RUN_DIR.parent / 'segnext_t_pretrained.pth'
    print('Downloading official MSCAN-T pretrained checkpoint from Tsinghua Cloud (can be slow, ~50MB)...')
    # urlretrieve has no timeout of its own -- a slow/stalled foreign host could hang the
    # whole session indefinitely. Read in chunks against an explicit per-request timeout instead,
    # so a bad connection fails fast into the from-scratch fallback rather than hanging.
    with urllib.request.urlopen(PRETRAINED_URL, timeout=120) as response, open(ckpt_path, 'wb') as f:
        shutil.copyfileobj(response, f)
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = state_dict.get('state_dict', state_dict)
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    if not backbone_state:
        raise RuntimeError('Downloaded checkpoint had no backbone.* keys -- unexpected format.')
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f'Loaded pretrained backbone. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
    if len(missing) > 5:
        # A handful of missing keys (e.g. layer_scale params not present in the ADE20K
        # checkpoint) is expected; a large number means the key names didn't really match
        # and this "success" would be misleading, so don't report it as loaded.
        raise RuntimeError(f'{len(missing)} missing keys is more than expected -- treating as a load failure rather than reporting a false positive.')
    pretrained_loaded = True
except Exception as exc:
    print(f'Pretrained backbone download/load failed ({exc}); continuing with the paper\'s from-scratch initialization scheme. This is not silent -- pretrained_backbone_loaded=False is saved in run_metadata.json.')

wandb.config.update({'pretrained_backbone_loaded': pretrained_loaded})
print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# ============================================================
# Train + evaluate. Same loop as the notebook; only the final save step
# changed (RunPod paths instead of Kaggle's, no display() calls).
# ============================================================
@torch.inference_mode()
def evaluate(loader, measure_inference=False):
    model.eval()
    true_positive = false_positive = false_negative = 0
    inference_seconds = 0.0
    image_count = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        if measure_inference:
            torch.cuda.synchronize()
            start = time.perf_counter()
        logits = model(pixel_values)
        if measure_inference:
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start

        predictions = logits.argmax(dim=1)
        true_positive += int(((predictions == 1) & (labels == 1)).sum().item())
        false_positive += int(((predictions == 1) & (labels == 0)).sum().item())
        false_negative += int(((predictions == 0) & (labels == 1)).sum().item())
        image_count += labels.shape[0]

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        # Pixel-level FPs per image, not per-instance -- see the notebook's
        # matching cell for why (no discrete detections to count over here).
        "fp_per_image": false_positive / max(image_count, 1),
        "inference_time_ms_per_image": 1000 * inference_seconds / max(image_count, 1),
        "images": image_count,
    }


RUN_DIR.mkdir(parents=True, exist_ok=True)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda", enabled=True)

history = []
best_dice = -1.0
best_epoch = -1
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.item())

    val_metrics = evaluate(val_loader)
    row = {"epoch": epoch, "train_loss": running_loss / len(train_loader), **val_metrics}
    history.append(row)
    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={row['train_loss']:.4f} | val Dice={row['dice']:.4f} | val IoU={row['iou']:.4f} | val Recall={row['recall']:.4f}")
    wandb.log(row, step=epoch)

    if row["dice"] > best_dice:
        best_dice = row["dice"]
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, RUN_DIR / "best_model.pt")
    else:
        epochs_without_improvement += 1

    pd.DataFrame(history).to_csv(RUN_DIR / "training_history.csv", index=False)
    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}; best validation Dice was {best_dice:.4f} at epoch {best_epoch}.")
        break

wandb.summary["best_epoch"] = best_epoch
wandb.summary["best_validation_dice"] = best_dice
print("Best epoch:", best_epoch, "Best validation Dice:", best_dice)

# ============================================================
# Load the best validation-Dice checkpoint and evaluate every fixed test subset.
# ============================================================
checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

test_rows = []
for split_name, split_samples in test_sets.items():
    metrics = evaluate(make_loader(split_samples), measure_inference=True)
    test_rows.append({"split": split_name, **metrics})
    print(split_name, metrics)
    wandb.log({f"test_{split_name}_{key}": value for key, value in metrics.items()})

test_df = pd.DataFrame(test_rows)
print(test_df.to_string(index=False))

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(RUN_DIR / "best_model.pt", FINAL_OUTPUT_DIR / "best_model.pt")
shutil.copy2(RUN_DIR / "training_history.csv", FINAL_OUTPUT_DIR / "training_history.csv")
# Long-format, one row per split -- same shape as D-FINE's evaluation_metrics.csv.
test_df.to_csv(FINAL_OUTPUT_DIR / "evaluation_metrics.csv", index=False)

by_split = {row["split"]: row for row in test_rows}
overall = by_split["overall"]

# Wide-format, one row per experiment -- matches the D-FINE/detection summary.csv
# schema exactly (see feedback_segmentation_metrics_schema in project memory for
# why mAP is intentionally left blank rather than fabricated for a dense segmenter).
summary_row = {
    "Experiment": RUN_NAME,
    "Model": MODEL_LABEL,
    "Batch": BATCH_SIZE,
    "Epochs": best_epoch,
    "mAP50": float("nan"),
    "mAP50_95": float("nan"),
    "Precision": overall["precision"],
    "Recall": overall["recall"],
    "mAP50_Small": float("nan"),
    "mAP50_Medium": float("nan"),
    "mAP50_Large": float("nan"),
    "Recall_Small": by_split["small"]["recall"],
    "Recall_Medium": by_split["medium"]["recall"],
    "Recall_Large": by_split["large"]["recall"],
    "Inference_Time_ms": overall["inference_time_ms_per_image"],
    "FP_per_Image": overall["fp_per_image"],
    "Dice": overall["dice"],
    "IoU": overall["iou"],
    "Notes": f"{MODEL_LABEL} via official Visual-Attention-Network/SegNeXt architecture (vendored, no MMSegmentation dependency) on RunPod, pretrained_backbone_loaded={pretrained_loaded}, binary defect segmentation, fixed 640 split. mAP intentionally blank -- not a valid metric for dense semantic segmentation.",
}
summary_df = pd.DataFrame([summary_row])
summary_df.to_csv(FINAL_OUTPUT_DIR / "summary.csv", index=False)

metadata = {
    "experiment": RUN_NAME,
    "model": MODEL_LABEL,
    "task": "binary semantic defect segmentation",
    "pretrained_backbone_loaded": pretrained_loaded,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "best_epoch": best_epoch,
    "best_validation_dice": best_dice,
    "split_counts": {
        "train": len(train_samples), "val": len(val_samples),
        "test": len(test_samples), "test_small": len(test_sets["small"]),
        "test_medium": len(test_sets["medium"]), "test_large": len(test_sets["large"]),
    },
}
(FINAL_OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

print("Saved final artifacts to:", FINAL_OUTPUT_DIR)
print(summary_df.to_string(index=False))
wandb.finish()
print("DONE.")
