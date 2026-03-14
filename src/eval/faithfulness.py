import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# Create grid masks used for insertion and deletion.
def make_grid_masks(H: int, W: int, k: int, device: torch.device):
    """Return list of [1,1,H,W] masks for k*k grid cells."""
    masks = []
    hs = np.linspace(0, H, k + 1).astype(int)
    ws = np.linspace(0, W, k + 1).astype(int)
    for i in range(k):
        for j in range(k):
            m = torch.zeros(1, 1, H, W, device=device)
            m[:, :, hs[i]:hs[i+1], ws[j]:ws[j+1]] = 1.0
            masks.append(m)
    return masks


def _is_grayscale_repeated(x: torch.Tensor, eps: float = 1e-6) -> bool:
    # x: [1,C,H,W]
    if x.shape[1] != 3:
        return False
    return ((x[:, 0] - x[:, 1]).abs().max() < eps) and ((x[:, 0] - x[:, 2]).abs().max() < eps)

def _odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)

# Expand a channel mean tensor to image shape.
def _expand_mean(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    # mean can be scalar, (C,), or (1,C,1,1); return broadcastable [1,C,1,1]
    if not torch.is_tensor(mean):
        mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    mean = mean.to(device=x.device, dtype=x.dtype)
    if mean.ndim == 0:
        return mean.view(1, 1, 1, 1)
    if mean.ndim == 1:
        if mean.numel() == x.shape[1]:
            return mean.view(1, x.shape[1], 1, 1)
        if mean.numel() >= 1 and x.shape[1] == 1:
            return mean.view(-1)[0].view(1, 1, 1, 1)
    if mean.ndim == 4:
        return mean
    # Fallback: use global mean scalar
    return mean.mean().view(1, 1, 1, 1)


# Create the baseline image used for masking.
def mask_fill_image(x: torch.Tensor, fill: str = "blur", mean: Optional[torch.Tensor] = None):
    """
    x: [1,C,H,W] in pixel space [0,1]
    """
    if fill == "zero":
        return torch.zeros_like(x)

    is_gray = _is_grayscale_repeated(x)
    _, _, H, W = x.shape

    if fill == "mean":
        if mean is not None:
            mean_bc = _expand_mean(x, mean)
            return mean_bc.expand_as(x)
        if is_gray:
            # Use a stable neutral gray baseline for grayscale-style inputs
            return torch.full_like(x, 0.5)
        # RGB histology: keep per-image mean
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x)

    if fill == "blur":
        # Make blur scale with resolution (works for all datasets)
        k = _odd(max(11, int(round(min(H, W) / 10))))   # ~23 for 224x224
        sigma = float(k) / 3.0                         # ~7–8 when k~23
        return gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])

    raise ValueError(f"Unknown fill: {fill}")

@torch.no_grad()
def score_of_target(model, x, target: int, score_type: str = "prob"):
    logits = model(x)[0]
    if score_type == "prob":
        probs = F.softmax(logits, dim=0)
        return float(probs[target].item())
    if score_type == "logit":
        return float(logits[target].item())
    if score_type == "margin":
        other_max = float(torch.max(torch.cat([logits[:target], logits[target+1:]])).item())
        return float(logits[target].item() - other_max)
    raise ValueError("score_type must be one of: prob, logit, margin")

# Compute insertion and deletion score curves.
def insertion_deletion_auc(
    model,
    x: torch.Tensor,
    target: int,
    sal_rank: torch.Tensor,       # [1,1,H,W] nonnegative ranking map
    mode: str = "deletion",       # "deletion" or "insertion"
    fill: str = "blur",
    mean: Optional[torch.Tensor] = None,
    score_type: str = "prob",
    max_pct: int = 45,
    step: int = 5,
):
    """
    Returns (auc, curve_probs) following pixel-level deletion/insertion
    with 5% increments up to 45% as described in the paper.
    """
    model.eval()
    device = x.device
    _, _, H, W = x.shape

    base = mask_fill_image(x, fill=fill, mean=mean)

    if mode == "deletion":
        src = base
    elif mode == "insertion":
        src = x
    else:
        raise ValueError("mode must be deletion or insertion")

    # Rank pixels by provided nonnegative ranking map.
    sal = sal_rank.reshape(-1)
    order = torch.argsort(sal, descending=True)

    n_total = int(sal.numel())
    steps_pct = list(range(0, max_pct + 1, step))  # 0,step,...,max_pct
    steps_n = [int(np.ceil(p / 100.0 * n_total)) for p in steps_pct]

    probs = []
    mask_flat = torch.zeros(n_total, device=device)
    prev_n = 0
    for n in steps_n:
        if n > prev_n:
            sel = order[prev_n:n]
            mask_flat[sel] = 1.0
            prev_n = n

        m = mask_flat.view(1, 1, H, W)
        if mode == "deletion":
            cur = x * (1 - m) + src * m
        else:  # insertion
            cur = base * (1 - m) + src * m

        probs.append(score_of_target(model, cur, target, score_type=score_type))

    y = np.array(probs, dtype=float)
    x_steps = np.array(steps_pct, dtype=float) / 100.0
    auc = float(np.trapz(y, x_steps))
    return auc, y


@torch.no_grad()
# Compute insertion and deletion accuracy curves.
def insertion_deletion_acc_auc(
    model,
    x: torch.Tensor,
    y_true: int,
    sal_rank: torch.Tensor,       # [1,1,H,W] nonnegative ranking map
    mode: str = "deletion",
    fill: str = "blur",
    mean: Optional[torch.Tensor] = None,
    max_pct: int = 45,
    step: int = 5,
):
    """
    Returns (auc, curve_acc) for accuracy over deletion/insertion steps.
    """
    model.eval()
    device = x.device
    _, _, H, W = x.shape

    base = mask_fill_image(x, fill=fill, mean=mean)

    if mode == "deletion":
        src = base
    elif mode == "insertion":
        src = x
    else:
        raise ValueError("mode must be deletion or insertion")

    sal = sal_rank.reshape(-1)
    order = torch.argsort(sal, descending=True)

    n_total = int(sal.numel())
    steps_pct = list(range(0, max_pct + 1, step))  # 0,step,...,max_pct
    steps_n = [int(np.ceil(p / 100.0 * n_total)) for p in steps_pct]

    accs = []
    mask_flat = torch.zeros(n_total, device=device)
    prev_n = 0
    for n in steps_n:
        if n > prev_n:
            sel = order[prev_n:n]
            mask_flat[sel] = 1.0
            prev_n = n

        m = mask_flat.view(1, 1, H, W)
        if mode == "deletion":
            cur = x * (1 - m) + src * m
        else:  # insertion
            cur = base * (1 - m) + src * m

        pred = int(model(cur)[0].argmax().item())
        accs.append(1.0 if pred == int(y_true) else 0.0)

    y = np.array(accs, dtype=float)
    x_steps = np.array(steps_pct, dtype=float) / 100.0
    auc = float(np.trapz(y, x_steps))
    return auc, y
