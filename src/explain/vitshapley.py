from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vit_masking import apply_patch_mask

ScoreMode = Literal["logit", "prob"]


@dataclass
class ViTShapleyConfig:
    score_mode: ScoreMode = "logit"
    normalize: bool = True  # min-max normalize for visualization
    efficiency_normalize: bool = True


# Predict patch-level ViTShapley attributions.
class ViTShapleyExplainer(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1, num_classes: int = 1):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be >= 1")
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, N, C] -> [B, N, num_classes]
        return self.mlp(patch_tokens)


# Convert logits into the requested target score.
def _score(logits: torch.Tensor, target: torch.Tensor, mode: ScoreMode) -> torch.Tensor:
    if mode == "logit":
        return logits.gather(1, target.view(-1, 1)).squeeze(1)
    if mode == "prob":
        probs = torch.softmax(logits, dim=1)
        return probs.gather(1, target.view(-1, 1)).squeeze(1)
    raise ValueError(f"Unknown score_mode: {mode}")

# Expand the target label to the batch shape.
def _ensure_target(target: torch.Tensor | int, b: int, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(target):
        tgt = target.to(device=device)
        if tgt.ndim == 0:
            tgt = tgt.view(1)
    else:
        tgt = torch.tensor([int(target)], device=device)
    if tgt.numel() == 1 and b > 1:
        tgt = tgt.expand(b)
    return tgt.view(-1)


# Rescale attributions to match the total score change.
def _efficiency_normalize(attrs: torch.Tensor, target_total: torch.Tensor) -> torch.Tensor:
    """
    Additive normalization so sum_i phi_i = target_total per sample.
    attrs: [B,N] or [B,N,C]
    target_total: [B] or [B,C]
    """
    if attrs.ndim == 2:
        diff = (target_total - attrs.sum(dim=1)) / max(1, attrs.shape[1])
        return attrs + diff.unsqueeze(1)
    if attrs.ndim == 3:
        diff = (target_total - attrs.sum(dim=1)) / max(1, attrs.shape[1])
        return attrs + diff.unsqueeze(1)
    raise ValueError(f"Unexpected attrs shape: {tuple(attrs.shape)}")


# Compute the training loss for the ViTShapley explainer.
def vitshapley_loss(
    model,
    explainer: ViTShapleyExplainer,
    x: torch.Tensor,
    target: torch.Tensor,
    token_keep_mask: torch.Tensor,
    num_prefix_tokens: int,
    score_mode: ScoreMode = "logit",
    score_model: Optional[nn.Module] = None,
    l2_lambda: float = 0.0,
) -> torch.Tensor:
    """
    Shapley regression loss (paper-style):
      sum_{i in S} a_i ~= f(x_S) - f(x_black)
    """
    scorer = score_model if score_model is not None else model
    target = _ensure_target(target, x.shape[0], x.device)

    # mask patches by replacing masked patches with black pixels
    x_masked = apply_patch_mask(x, model.backbone, token_keep_mask, fill=0.0)
    logits_masked = scorer(x_masked)
    f_mask = _score(logits_masked, target, score_mode)

    x_black = torch.zeros_like(x)
    logits_black = scorer(x_black)
    f_black = _score(logits_black, target, score_mode)

    logits_full = scorer(x)
    f_full = _score(logits_full, target, score_mode)

    # explainer outputs on full tokens
    tokens = model.extract_tokens(x)
    patch_tokens = tokens[:, num_prefix_tokens:, :]
    attrs = explainer(patch_tokens)
    if attrs.ndim == 2:
        attrs = attrs.unsqueeze(-1)
    # select target class
    tgt = target.view(-1, 1, 1).expand(-1, attrs.shape[1], 1)
    attrs = attrs.gather(-1, tgt).squeeze(-1)

    mask_patches = token_keep_mask[:, num_prefix_tokens:].to(attrs.dtype)
    if mask_patches.shape != attrs.shape:
        raise ValueError("Mask/attr shape mismatch.")

    attrs = _efficiency_normalize(attrs, (f_full - f_black))
    pred_sum = (attrs * mask_patches).sum(dim=1)
    loss = F.mse_loss(pred_sum, (f_mask - f_black))

    if l2_lambda and l2_lambda > 0.0:
        loss = loss + l2_lambda * (attrs ** 2).mean()
    return loss


@torch.no_grad()
# Compute a ViTShapley attribution map.
def vitshapley_attribution(
    model,
    x: torch.Tensor,
    explainer: ViTShapleyExplainer,
    num_prefix_tokens: int,
    target: Optional[int | torch.Tensor] = None,
    cfg: ViTShapleyConfig = ViTShapleyConfig(),
) -> torch.Tensor:
    """
    Returns [B,1,H,W] attribution map from explainer output.
    """
    tokens = model.extract_tokens(x)
    patch_tokens = tokens[:, num_prefix_tokens:, :]
    attrs = explainer(patch_tokens)
    if attrs.ndim == 2:
        attrs = attrs.unsqueeze(-1)

    # pick target class
    logits_full = None
    if target is None:
        logits_full = model(x)
        target = logits_full.argmax(dim=1)
    target = _ensure_target(target, x.shape[0], x.device)
    tgt = target.view(-1, 1, 1).expand(-1, attrs.shape[1], 1)
    attrs = attrs.gather(-1, tgt).squeeze(-1)  # [B, N]

    if cfg.efficiency_normalize:
        if logits_full is None:
            logits_full = model(x)
        logits_black = model(torch.zeros_like(x))
        f_full = _score(logits_full, target, cfg.score_mode)
        f_black = _score(logits_black, target, cfg.score_mode)
        attrs = _efficiency_normalize(attrs, (f_full - f_black))

    # reshape to grid
    b, n = attrs.shape
    h = x.shape[-2]
    w = x.shape[-1]
    # infer grid from num patches
    grid = int(n ** 0.5)
    if grid * grid != n:
        raise ValueError(f"Cannot reshape {n} patches into square grid.")

    ph = h // grid
    pw = w // grid
    patch_map = attrs.view(b, grid, grid)
    full = patch_map.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2)
    full = full.unsqueeze(1)

    if cfg.normalize:
        vmin = full.amin(dim=(1, 2, 3), keepdim=True)
        vmax = full.amax(dim=(1, 2, 3), keepdim=True)
        denom = (vmax - vmin).clamp_min(1e-12)
        full = (full - vmin) / denom
    return full
