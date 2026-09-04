"""
Dataset for the auxiliary prompt-grounding task: pairs a cropped defect region
with one of its real referring-expression descriptions (defect_prompts_v2.json).

Purely a data-loading component -- independent of whichever backbone / cross-attention
design the main detection+segmentation model ends up using, so this can be built and
verified before those choices are finalized.

Training-only. Nothing here runs at inference -- the point of the auxiliary task this
feeds is to shape the shared backbone's features, then be discarded.
"""
import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

SIZE_BUCKETS = ['small', 'medium', 'large']

# p1-p6 are single strings; p7-p9 are lists of 5 alternative phrasings (same fact,
# different wording) -- sampling a random variant is free augmentation on the text
# side. p0 is always empty in the source schema (unused placeholder), skipped.
SINGLE_LEVELS = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
LIST_LEVELS = ['p7', 'p8', 'p9']


def sample_prompt(prompts, rng):
    """Uniform over all 9 levels for now, then a random variant if that level is a
    list. A placeholder policy, not a final one -- e.g. weighting toward p8/p9 for
    instances with domain-documented severity (real criticality info, not a pixel-
    derived score) is worth revisiting once there's training signal to judge it by."""
    level = rng.choice(SINGLE_LEVELS + LIST_LEVELS)
    value = prompts[level]
    return rng.choice(value) if isinstance(value, list) else value


def resolve_image_path(dataset_root, dataset, img_name):
    for size in SIZE_BUCKETS:
        path = dataset_root / dataset / size / 'images' / img_name
        if path.exists():
            return path
    return None


class DefectPromptPairDataset(Dataset):
    """Yields (image_crop, prompt_text) pairs for the region-text alignment
    auxiliary task. One __getitem__ call = one defect instance, one sampled
    prompt from its real p1-p9 hierarchy.

    Returns a raw PIL crop and a raw string, not a preprocessed tensor -- the actual
    resize/normalize should match whatever image encoder the main model ends up
    using, which isn't decided yet. Tensor-ification belongs in a collate_fn added
    once that choice is made, not baked in here.
    """

    def __init__(self, prompts_json_path, dataset_root, crop_pad_ratio=0.3, seed=42):
        self.dataset_root = Path(dataset_root)
        self.crop_pad_ratio = crop_pad_ratio
        self.rng = random.Random(seed)

        with open(prompts_json_path, encoding='utf-8') as f:
            raw_entries = json.load(f)

        self.entries = []
        skipped = 0
        for entry in raw_entries:
            path = resolve_image_path(self.dataset_root, entry['dataset'], entry['img_name'])
            if path is None:
                skipped += 1
                continue
            self.entries.append({**entry, '_image_path': path})
        self.skipped = skipped

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        image = Image.open(entry['_image_path']).convert('RGB')
        w, h = image.size

        x1, y1, x2, y2 = entry['bbox']
        bw, bh = x2 - x1, y2 - y1
        pad_x = max(int(bw * self.crop_pad_ratio), 4)
        pad_y = max(int(bh * self.crop_pad_ratio), 4)
        cx1, cy1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
        cx2, cy2 = min(x2 + pad_x, w), min(y2 + pad_y, h)
        crop = image.crop((cx1, cy1, cx2, cy2))

        prompt_text = sample_prompt(entry['prompts'], self.rng)

        return {
            'image_crop': crop,
            'prompt_text': prompt_text,
            'segment_id': entry['segment_id'],
            'dataset': entry['dataset'],
            'severity_source': entry.get('severity_source'),
        }


if __name__ == '__main__':
    import sys

    prompts_path = sys.argv[1] if len(sys.argv) > 1 else 'defect_prompts_v2.json'
    dataset_root = sys.argv[2] if len(sys.argv) > 2 else 'processed_output'

    ds = DefectPromptPairDataset(prompts_json_path=prompts_path, dataset_root=dataset_root)
    print(f'Loaded {len(ds)} instances, skipped {ds.skipped} with unresolved image paths.')

    rng = random.Random(0)
    for i in rng.sample(range(len(ds)), k=5):
        sample = ds[i]
        print(f"[{sample['dataset']}] {sample['segment_id']}")
        print(f"  crop size: {sample['image_crop'].size}  severity_source: {sample['severity_source']}")
        print(f"  prompt: {sample['prompt_text']}")
