import torch

# Reduce an attribution tensor to one saliency map.
def reduce_attribution(attr: torch.Tensor) -> torch.Tensor:
    """
    Signed reduction to [1,1,H,W] without destroying sign.
    """
    if attr.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {attr.shape}")
    if attr.shape[1] == 1:
        return attr
    return attr.sum(dim=1, keepdim=True)

# Convert a signed map into a ranking map.
def saliency_for_ranking(sal_signed: torch.Tensor, mode: str = "magnitude") -> torch.Tensor:
    """
    Nonnegative ranking map [1,1,H,W].
    """
    if mode == "magnitude":
        return sal_signed.abs()
    if mode == "positive":
        return sal_signed.clamp_min(0)
    if mode == "negative":
        return (-sal_signed).clamp_min(0)
    raise ValueError(f"Unknown saliency mode: {mode}")

# Normalize a saliency map.
def normalize_map(m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a [1,1,H,W] map to [0,1] per-image.
    """
    mn = m.amin(dim=(-2, -1), keepdim=True)
    mx = m.amax(dim=(-2, -1), keepdim=True)
    return (m - mn) / (mx - mn + eps)
