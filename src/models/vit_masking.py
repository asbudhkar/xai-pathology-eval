from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
# Store the token mask and metadata for masked ViT runs.
class MaskSpec:
    token_keep_mask: torch.Tensor  # [B, N] bool, True = keep
    n_prefix: int
    n_patches: int


# Read the patch size from a ViT backbone.
def _get_patch_size(backbone) -> Tuple[int, int]:
    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is None:
        raise ValueError("Backbone missing patch_embed; cannot infer patch size.")
    ps = getattr(patch_embed, "patch_size", None)
    if ps is None:
        raise ValueError("patch_embed missing patch_size.")
    if isinstance(ps, (list, tuple)):
        return int(ps[0]), int(ps[1])
    return int(ps), int(ps)


# Return the number of prefix tokens in a backbone.
def num_prefix_tokens(backbone) -> int:
    # Prefer explicit prefix token count; fall back to reg_tokens for UNI-style models.
    n = getattr(backbone, "num_prefix_tokens", None)
    if n is not None:
        return int(n)
    reg = getattr(backbone, "reg_tokens", None)
    if reg is not None:
        return int(reg)
    return 1


# Sample a random token keep mask.
def build_token_keep_mask(
    x: torch.Tensor,
    backbone,
    ratio_min: float,
    ratio_max: float,
    generator: Optional[torch.Generator] = None,
) -> MaskSpec:
    """
    Build per-example token keep mask.
    Keep all prefix tokens (CLS/reg); mask only patch tokens.
    Mask ratio sampled uniformly in [ratio_min, ratio_max] per example.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")
    b, _, h, w = x.shape
    ph, pw = _get_patch_size(backbone)
    if h % ph != 0 or w % pw != 0:
        raise ValueError(f"H,W must be divisible by patch size. H={h} W={w} ps=({ph},{pw})")

    n_patches = (h // ph) * (w // pw)
    n_prefix = num_prefix_tokens(backbone)

    if generator is None:
        generator = torch.Generator(device=x.device)
    r = torch.empty(b, device=x.device).uniform_(ratio_min, ratio_max, generator=generator)
    n_mask = torch.round(r * n_patches).to(torch.long)
    # If no prefix tokens are kept, ensure at least one patch token stays unmasked.
    if n_prefix == 0 and n_patches > 0:
        n_mask = torch.clamp(n_mask, max=n_patches - 1)

    keep_mask = torch.ones((b, n_prefix + n_patches), device=x.device, dtype=torch.bool)
    if n_patches == 0:
        return MaskSpec(keep_mask, n_prefix, n_patches)

    for i in range(b):
        k = int(n_mask[i].item())
        if k <= 0:
            continue
        idx = torch.randperm(n_patches, device=x.device, generator=generator)[:k]
        keep_mask[i, n_prefix + idx] = False

    keep_counts = keep_mask.sum(dim=1)
    if torch.any(keep_counts == 0):
        n_zero = int((keep_counts == 0).sum().item())
        print(
            f"[vit_masking] WARNING: {n_zero}/{b} samples have 0 kept tokens "
            f"(n_prefix={n_prefix}, n_patches={n_patches}, ratio_min={ratio_min}, ratio_max={ratio_max})"
        )

    return MaskSpec(keep_mask, n_prefix, n_patches)


# Sample token masks with matched cardinality.
def build_token_keep_mask_uniform_cardinality(
    x: torch.Tensor,
    backbone,
    generator: Optional[torch.Generator] = None,
) -> MaskSpec:
    """
    Build per-example token keep mask by sampling a cardinality k uniformly,
    then selecting k patch tokens to keep. Prefix tokens are always kept.
    This matches the paper's uniform-by-cardinality mask distribution.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")
    b, _, h, w = x.shape
    ph, pw = _get_patch_size(backbone)
    if h % ph != 0 or w % pw != 0:
        raise ValueError(f"H,W must be divisible by patch size. H={h} W={w} ps=({ph},{pw})")

    n_patches = (h // ph) * (w // pw)
    n_prefix = num_prefix_tokens(backbone)
    keep_mask = torch.zeros((b, n_prefix + n_patches), device=x.device, dtype=torch.bool)
    keep_mask[:, :n_prefix] = True
    if n_patches == 0:
        return MaskSpec(keep_mask, n_prefix, n_patches)

    if generator is None:
        generator = torch.Generator(device=x.device)

    # Sample k ~ Uniform({0..n_patches}) per example, then keep k patches.
    k = torch.randint(0, n_patches + 1, (b,), device=x.device, generator=generator)
    for i in range(b):
        ki = int(k[i].item())
        if ki <= 0:
            continue
        idx = torch.randperm(n_patches, device=x.device, generator=generator)[:ki]
        keep_mask[i, n_prefix + idx] = True

    return MaskSpec(keep_mask, n_prefix, n_patches)


# Apply a patch mask directly to the input image.
def apply_patch_mask(
    x: torch.Tensor,
    backbone,
    token_keep_mask: torch.Tensor,
    fill: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """
    Apply a patch keep mask to an image by replacing masked patches with `fill`.
    token_keep_mask: [B, N] bool, True = keep (includes prefix tokens).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")
    b, c, h, w = x.shape
    ph, pw = _get_patch_size(backbone)
    if h % ph != 0 or w % pw != 0:
        raise ValueError(f"H,W must be divisible by patch size. H={h} W={w} ps=({ph},{pw})")

    gh, gw = h // ph, w // pw
    n_patches = gh * gw
    n_prefix = num_prefix_tokens(backbone)
    if token_keep_mask.shape[1] != n_prefix + n_patches:
        raise ValueError(
            f"token_keep_mask has {token_keep_mask.shape[1]} tokens, expected {n_prefix + n_patches}."
        )

    patch_keep = token_keep_mask[:, n_prefix:].view(b, gh, gw)
    patch_keep = patch_keep.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2)
    patch_keep = patch_keep.unsqueeze(1).to(x.dtype)  # [B,1,H,W]

    if isinstance(fill, torch.Tensor):
        fill_t = fill.to(device=x.device, dtype=x.dtype)
    else:
        fill_t = torch.tensor(fill, device=x.device, dtype=x.dtype)

    if fill_t.ndim == 0:
        fill_t = fill_t.view(1, 1, 1, 1)
    return x * patch_keep + fill_t * (1 - patch_keep)


# Build a mask that keeps all image patches.
def build_empty_token_keep_mask(x: torch.Tensor, backbone) -> MaskSpec:
    """Keep only prefix tokens; mask all patch tokens."""
    if x.ndim != 4:
        raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")
    b, _, h, w = x.shape
    ph, pw = _get_patch_size(backbone)
    n_patches = (h // ph) * (w // pw)
    n_prefix = num_prefix_tokens(backbone)
    keep_mask = torch.zeros((b, n_prefix + n_patches), device=x.device, dtype=torch.bool)
    keep_mask[:, :n_prefix] = True
    return MaskSpec(keep_mask, n_prefix, n_patches)


# Patch attention forward to capture attention weights.
def _patch_attention_forward(attn_module: torch.nn.Module, token_keep_mask: torch.Tensor):
    # Handle the forward step.
    def forward(x, **kwargs):
        attn_mask = kwargs.get("attn_mask", None)
        b, n, c = x.shape
        qkv = attn_module.qkv(x)
        qkv = qkv.reshape(b, n, 3, attn_module.num_heads, c // attn_module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if hasattr(attn_module, "q_norm") and attn_module.q_norm is not None:
            q = attn_module.q_norm(q)
        if hasattr(attn_module, "k_norm") and attn_module.k_norm is not None:
            k = attn_module.k_norm(k)

        scale = getattr(attn_module, "scale", None)
        if scale is None:
            scale = (q.shape[-1] ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale

        # token_keep_mask: [B, N] True=keep
        if token_keep_mask is not None:
            km = token_keep_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
            attn = attn.masked_fill(~km, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        if hasattr(attn_module, "attn_drop") and attn_module.attn_drop is not None:
            attn = attn_module.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        proj = getattr(attn_module, "proj", None)
        if proj is not None:
            x_out = proj(x_out)
        proj_drop = getattr(attn_module, "proj_drop", None)
        if proj_drop is not None:
            x_out = proj_drop(x_out)

        # Zero out masked query tokens to prevent leakage
        if token_keep_mask is not None:
            x_out = x_out * token_keep_mask.unsqueeze(-1).to(x_out.dtype)
        return x_out
    return forward


# Patch a backbone so attention respects token masks.
class AttnMaskPatcher:
    """
    Temporarily patch backbone attention to respect a token keep mask.
    """
    def __init__(self, backbone: torch.nn.Module, token_keep_mask: torch.Tensor):
        self._backbone = backbone
        self._token_keep_mask = token_keep_mask
        self._orig = []

    def __enter__(self):
        blocks = getattr(self._backbone, "blocks", None)
        if blocks is None:
            return self
        for blk in blocks:
            if not hasattr(blk, "attn"):
                continue
            attn = blk.attn
            self._orig.append((attn, attn.forward))
            attn.forward = _patch_attention_forward(attn, self._token_keep_mask)
        return self

    def __exit__(self, exc_type, exc, tb):
        for attn, orig_fwd in self._orig:
            attn.forward = orig_fwd
        self._orig = []
        return False


@contextlib.contextmanager
# Apply masked attention patches inside a context manager.
def masked_attention(backbone: torch.nn.Module, token_keep_mask: torch.Tensor):
    patcher = AttnMaskPatcher(backbone, token_keep_mask)
    try:
        patcher.__enter__()
        yield
    finally:
        patcher.__exit__(None, None, None)
