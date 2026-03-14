import torch
import numpy as np
from torchvision.transforms import ColorJitter
from src.explain.utils import normalize_map

# Compute correlation between two heatmaps.
def corr2d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    a,b: [1,1,H,W] in any range
    Returns Pearson correlation over pixels.
    """
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < eps:
        return 0.0
    return float((a @ b).item() / denom)

# Flip horizontally.
def hflip(x):  # x: [1,C,H,W]
    return torch.flip(x, dims=[3])

# Flip vertically.
def vflip(x):
    return torch.flip(x, dims=[2])

# Map a flipped heatmap back to the original orientation.
def unflip_map(m, mode):
    if mode == "h":
        return torch.flip(m, dims=[3])
    if mode == "v":
        return torch.flip(m, dims=[2])
    raise ValueError(mode)

# Color jitter transform.
def make_jitter():
    return ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01)

@torch.no_grad()
# Compare two explanations for one transformed sample.
def stability_one(
    sal_orig: torch.Tensor,      # [1,1,H,W]
    sal_trans: torch.Tensor,     # [1,1,H,W], already aligned to orig coords
) -> float:
    sal_o = normalize_map(sal_orig)
    sal_t = normalize_map(sal_trans)
    return corr2d(sal_o, sal_t)
