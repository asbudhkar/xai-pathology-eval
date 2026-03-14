import math
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# Normalize an attention matrix for rollout.
def _sanitize_attention(attn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not torch.isfinite(attn).all():
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    denom = attn.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > eps, denom, torch.ones_like(denom))
    return attn / denom


# Return the transformer blocks from a ViT backbone.
def _get_vit_blocks(backbone: torch.nn.Module) -> List[torch.nn.Module]:
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        return list(backbone.blocks)
    if hasattr(backbone, "layers") and len(backbone.layers) > 0:
        return list(backbone.layers)
    raise ValueError("Backbone has no transformer blocks/layers for attention rollout.")


# Estimate how many prefix tokens a ViT uses.
def _num_prefix_tokens(backbone: torch.nn.Module, n_tokens: int) -> int:
    prefix = getattr(backbone, "num_prefix_tokens", None)
    if prefix is None:
        return 1
    return int(prefix)


# Infer the patch grid size for a token map.
def _grid_size(backbone: torch.nn.Module, num_patches: int) -> Tuple[int, int]:
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "grid_size"):
        grid = backbone.patch_embed.grid_size
        if isinstance(grid, (tuple, list)) and len(grid) == 2:
            return int(grid[0]), int(grid[1])
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Cannot infer square grid from {num_patches} patches.")
    return side, side


# Patch attention forward to capture attention weights.
def _patch_attention_forward(attn_module: torch.nn.Module, store: List[torch.Tensor], use_grad: bool):
    if not hasattr(attn_module, "qkv"):
        raise ValueError("Attention module missing qkv projection; unsupported for rollout.")

    num_heads = int(getattr(attn_module, "num_heads", 0))
    if num_heads <= 0:
        raise ValueError("Attention module missing num_heads; unsupported for rollout.")

    proj = getattr(attn_module, "proj", None)
    proj_drop = getattr(attn_module, "proj_drop", None)
    attn_drop = getattr(attn_module, "attn_drop", None)
    q_norm = getattr(attn_module, "q_norm", None)
    k_norm = getattr(attn_module, "k_norm", None)

    def forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, n, c = x.shape
        head_dim = c // num_heads
        scale = getattr(attn_module, "scale", None)
        if scale is None:
            scale = head_dim ** -0.5

        qkv = attn_module.qkv(x)
        qkv = qkv.reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = _sanitize_attention(attn)
        if attn_drop is not None:
            attn = attn_drop(attn)

        if use_grad and attn.requires_grad:
            attn.retain_grad()
        store.append(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        if proj is not None:
            x = proj(x)
        if proj_drop is not None:
            x = proj_drop(x)
        return x

    return forward


# Define the AttnPatcher helper.
class _AttnPatcher:
    def __init__(self, backbone: torch.nn.Module, use_grad: bool):
        self._blocks = _get_vit_blocks(backbone)
        self._use_grad = use_grad
        self._store: List[torch.Tensor] = []
        self._orig = []

    def __enter__(self):
        for blk in self._blocks:
            if not hasattr(blk, "attn"):
                continue
            attn = blk.attn
            self._orig.append((attn, attn.forward))
            attn.forward = _patch_attention_forward(attn, self._store, self._use_grad)
        return self._store

    def __exit__(self, exc_type, exc, tb):
        for attn, orig_fwd in self._orig:
            attn.forward = orig_fwd
        return False


# Compute an attention rollout heatmap.
def attention_rollout_attribution(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: Optional[int] = None,
    use_grad: bool = True,
) -> torch.Tensor:
    """
    Attention rollout for ViT-style backbones.
    If use_grad=True, weights attention by gradients (Chefer-style).
    Returns [1,1,H,W] map aligned to input size.
    """
    if not hasattr(model, "backbone"):
        raise ValueError("Attention rollout requires a ViT-style backbone on model.backbone.")

    if use_grad and target is None:
        raise ValueError("use_grad=True requires a target class index.")

    backbone = model.backbone

    if use_grad:
        # Ensure a grad-tracked path even when the backbone is frozen.
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad() if use_grad else nullcontext():
        with _AttnPatcher(backbone, use_grad=use_grad) as attn_list:
            logits = model(x)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if use_grad:
        score = logits[0, int(target)]
        score.backward()

    if use_grad and any(a.grad is None for a in attn_list):
        return attention_rollout_attribution(model, x.detach(), target=target, use_grad=False)

    if len(attn_list) == 0:
        raise ValueError("No attention maps captured for rollout.")

    b, h, n, _ = attn_list[0].shape
    eye = torch.eye(n, device=attn_list[0].device).unsqueeze(0).expand(b, n, n)
    rollout = eye

    for attn in attn_list:
        if use_grad:
            if attn.grad is None:
                raise RuntimeError("use_grad=True but attention gradients are missing.")
            attn = attn * attn.grad
            attn = attn.clamp(min=0.0)
            attn_fused = attn.mean(dim=1)
        else:
            attn_fused = attn.mean(dim=1)

        if not torch.isfinite(attn_fused).all():
            attn_fused = torch.nan_to_num(attn_fused, nan=0.0, posinf=0.0, neginf=0.0)
        attn_fused = attn_fused + eye
        attn_fused = _sanitize_attention(attn_fused)
        rollout = attn_fused.bmm(rollout)

    prefix = _num_prefix_tokens(backbone, n)
    if prefix >= n:
        raise ValueError("Prefix tokens exceed attention size; cannot build rollout map.")

    mask = rollout[:, 0, prefix:]
    gh, gw = _grid_size(backbone, mask.shape[-1])
    mask = mask.view(b, 1, gh, gw)
    mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
    if not torch.isfinite(mask).all():
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    denom = mask.sum(dim=(-2, -1), keepdim=True)
    denom = torch.where(denom > 1e-6, denom, torch.ones_like(denom))
    mask = mask / denom
    if (mask.sum(dim=(-2, -1), keepdim=True) <= 1e-6).any():
        mask = mask + 1e-6
        mask = mask / mask.sum(dim=(-2, -1), keepdim=True)
    return mask.detach()
