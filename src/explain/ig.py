import torch
from torchvision.transforms.functional import gaussian_blur
from captum.attr import IntegratedGradients

# Compute an Integrated Gradients attribution map.
def ig_attribution(
    model,
    x: torch.Tensor,
    target: int,
    baseline: str = "mean",     # "mean", "zero", or "blur"
    n_steps: int = 32,
):
    """
    x: [1,C,H,W] on device
    returns attr: [1,C,H,W]
    """
    model.eval()
    x = x.float().requires_grad_(True)

    def _forward(inp: torch.Tensor) -> torch.Tensor:
        out = model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    if baseline == "zero":
        b = torch.zeros_like(x)
    elif baseline == "mean":
        b = torch.full_like(x, 0.5)
    elif baseline == "blur":
        b = gaussian_blur(x, kernel_size=[11, 11], sigma=2.0)
    elif baseline == "blur21":
        b = gaussian_blur(x, kernel_size=[21, 21], sigma=4.0)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    b = b.detach()
    ig = IntegratedGradients(_forward)
    attr = ig.attribute(x, baselines=b, target=target, n_steps=n_steps)
    return attr.detach()
