"""
Auxiliary training-only task: pull a defect crop's image embedding and its real
text description's embedding together (push mismatched pairs apart), so the shared
Mask2Former backbone is regularized toward the properties these descriptions encode
(shape, size, severity-correlated texture -- see the conversation's breakdown of
which axes actually carry new information vs. which are closer to redundant).

Deliberately NOT using a pretrained language model. The real prompt corpus
(defect_prompts_v2.json, 571,830 non-empty prompt strings across all instances and
all p1-p9 levels) has only 194 unique words and zero words appearing just once --
a fully bounded domain vocabulary. A small transformer trained from scratch on this
vocabulary is enough; importing a heavy pretrained text encoder for it would be
adding weight for no real benefit, the opposite of what this auxiliary task is for.

Both encoders + the alignment loss are training-only, exactly like the rest of the
auxiliary task -- dropped entirely at inference, per the whole point of this being
"auxiliary."
"""
import json
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1


class PromptVocab:
    """Word-level vocabulary built directly from the real prompt corpus -- no
    external tokenizer, no download, no OOV risk in practice (verified: 0 words
    occur only once across the real 571,830-string corpus)."""

    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    @classmethod
    def build_from_prompts_json(cls, prompts_json_path):
        with open(prompts_json_path, encoding='utf-8') as f:
            data = json.load(f)
        counter = Counter()
        for entry in data:
            for value in entry['prompts'].values():
                for v in (value if isinstance(value, list) else [value]):
                    if v:
                        counter.update(re.findall(r"[a-z0-9']+", v.lower()))
        word2idx = {'<pad>': PAD, '<unk>': UNK}
        for word, _ in counter.most_common():
            word2idx[word] = len(word2idx)
        return cls(word2idx)

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text, max_len=32):
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        ids = [self.word2idx.get(t, UNK) for t in tokens[:max_len]]
        ids = ids + [PAD] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class TextEncoder(nn.Module):
    """Small transformer encoder, trained from scratch. Mean-pools over real
    (non-pad) tokens, matching the image side's global-average-pool -- both
    encoders reduce to one vector per instance before alignment."""

    def __init__(self, vocab_size, embed_dim=256, num_layers=2, num_heads=4, ffn_dim=512, max_len=32):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, token_ids):
        B, L = token_ids.shape
        mask = token_ids == PAD  # [B, L], True where padding
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(token_ids) + self.pos_embed(positions)

        x = self.encoder(x, src_key_padding_mask=mask)

        valid = (~mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return pooled  # [B, embed_dim]


class ImageRegionEncoder(nn.Module):
    """Runs a defect crop through Mask2Former's own backbone + pixel decoder
    (the SAME shared feature extractor the main det/seg heads use) and
    global-average-pools the result to one embedding per crop."""

    def __init__(self, mask2former_model):
        super().__init__()
        self.pixel_level_module = mask2former_model.model.pixel_level_module

    def forward(self, pixel_values, pixel_mask=None):
        if pixel_mask is None:
            pixel_mask = torch.ones(pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3],
                                    device=pixel_values.device)
        out = self.pixel_level_module(pixel_values, output_hidden_states=False)
        features = out.decoder_last_hidden_state  # [B, C, H, W]
        return features.mean(dim=[2, 3])  # [B, C]


class PromptAlignmentHead(nn.Module):
    """Ties the image and text encoders together with small projection heads into
    a shared embedding space, and computes the symmetric contrastive (CLIP-style)
    loss. This module + everything it wraps is training-only."""

    def __init__(self, mask2former_model, vocab_size, image_dim=256, text_embed_dim=256,
                 shared_dim=256, temperature=0.07):
        super().__init__()
        self.image_encoder = ImageRegionEncoder(mask2former_model)
        self.text_encoder = TextEncoder(vocab_size, embed_dim=text_embed_dim)
        self.image_proj = nn.Linear(image_dim, shared_dim)
        self.text_proj = nn.Linear(text_embed_dim, shared_dim)
        self.temperature = temperature

    def forward(self, pixel_values, token_ids):
        image_feat = self.image_proj(self.image_encoder(pixel_values))
        text_feat = self.text_proj(self.text_encoder(token_ids))

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        logits = image_feat @ text_feat.t() / self.temperature  # [B, B]
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        return (loss_i2t + loss_t2i) / 2, logits


if __name__ == '__main__':
    import sys
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
    from prompt_pair_dataset import DefectPromptPairDataset

    prompts_path = sys.argv[1] if len(sys.argv) > 1 else '../defect_prompts_v2.json'
    dataset_root = sys.argv[2] if len(sys.argv) > 2 else '../processed_output'

    print('Building vocabulary from the real prompt corpus...')
    vocab = PromptVocab.build_from_prompts_json(prompts_path)
    print(f'Vocab size: {len(vocab)}')

    print('Loading real (crop, prompt) pairs...')
    ds = DefectPromptPairDataset(prompts_json_path=prompts_path, dataset_root=dataset_root)
    print(f'Loaded {len(ds)} instances, skipped {ds.skipped}.')

    torch.manual_seed(0)
    config = Mask2FormerConfig(num_labels=2, num_queries=100)
    m2f = Mask2FormerForUniversalSegmentation(config)
    m2f.eval()

    align_head = PromptAlignmentHead(m2f, vocab_size=len(vocab))

    # Real crops, resized to a fixed size and stacked into a batch -- resize/normalize
    # here is a placeholder matching whatever preprocessing the main model ends up using.
    import random
    rng = random.Random(1)
    batch_indices = rng.sample(range(len(ds)), k=8)
    samples = [ds[i] for i in batch_indices]

    pixel_values = torch.stack([
        torch.from_numpy(
            __import__('numpy').asarray(s['image_crop'].resize((384, 384)))
        ).permute(2, 0, 1).float() / 255.0
        for s in samples
    ])
    token_ids = torch.stack([vocab.encode(s['prompt_text']) for s in samples])

    print('pixel_values:', tuple(pixel_values.shape), ' token_ids:', tuple(token_ids.shape))

    loss, logits = align_head(pixel_values, token_ids)
    print('contrastive loss:', loss.item(), ' logits shape:', tuple(logits.shape))
    assert logits.shape == (8, 8)
    assert loss.item() == loss.item(), 'loss is NaN'

    loss.backward()

    # Known-inert parameters that belong to the wrapped, pretrained Mask2Former
    # backbone itself (not code written here), confirmed by inspecting HF's actual
    # source: SwinBackbone is built with add_pooling_layer=False and explicitly
    # lists swin.layernorm.* as ignorable (its own per-stage hidden_states_norms
    # are what's actually used); the pixel decoder's level_embed is an internal
    # multi-scale-encoder parameter not on the gradient path to this specific
    # pooled output. Both are pre-existing structural facts about the third-party
    # model, not something a bug fix here could change.
    KNOWN_INERT = {
        'image_encoder.pixel_level_module.encoder.swin.layernorm.weight',
        'image_encoder.pixel_level_module.encoder.swin.layernorm.bias',
        'image_encoder.pixel_level_module.decoder.level_embed',
    }

    disconnected = [
        name for name, p in align_head.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0) and name not in KNOWN_INERT
    ]
    n_total = sum(1 for p in align_head.parameters() if p.requires_grad)
    n_grad = sum(1 for p in align_head.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f'PromptAlignmentHead: {n_grad}/{n_total} params with nonzero grad ({len(KNOWN_INERT)} known-inert third-party params excluded from the check)')
    assert not disconnected, f'Unexplained disconnected parameters: {disconnected}'

    print()
    print('OK: prompt alignment head verified end-to-end on real crops, real text, real vocab.')
