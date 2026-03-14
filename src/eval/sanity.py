import copy
import torch
import numpy as np
from src.explain.utils import normalize_map

# Compute Pearson correlation on flattened tensors.
def pearson_corr_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < eps:
        return 0.0
    return float((a @ b).item() / denom)

# Randomize the weights of one module in place.
def randomize_module_weights_(m: torch.nn.Module):
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                torch.nn.init.zeros_(mod.bias)
        elif isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
            if hasattr(mod, "weight") and mod.weight is not None:
                torch.nn.init.ones_(mod.weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                torch.nn.init.zeros_(mod.bias)

def _count_params(mod: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in mod.parameters()))

# Return the classifier head module.
def _get_head_module(model: torch.nn.Module):
    if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Module):
        return model.classifier
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        return model.fc
    return None

# List backbone blocks in model order.
def _ordered_backbone_blocks(model: torch.nn.Module):
    backbone = getattr(model, "backbone", model)

    # ResNet-18: stem -> layer1 -> layer2 -> layer3 -> layer4
    if all(hasattr(backbone, k) for k in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]):
        blocks = [("stem", torch.nn.Sequential(backbone.conv1, backbone.bn1))]
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(backbone, lname)
            for i, blk in enumerate(layer):
                blocks.append((f"{lname}.{i}", blk))
        return blocks

    # EfficientNet-B0 (torchvision): features is sequential; treat each stage as a block group.
    if hasattr(backbone, "features") and isinstance(backbone.features, torch.nn.Sequential):
        return [(f"features.{i}", blk) for i, blk in enumerate(backbone.features)]

    # ViT backbones (timm): patch embed -> blocks -> norm
    if hasattr(backbone, "blocks"):
        blocks = []
        if hasattr(backbone, "patch_embed"):
            blocks.append(("patch_embed", backbone.patch_embed))
        for i, blk in enumerate(backbone.blocks):
            blocks.append((f"blocks.{i}", blk))
        if hasattr(backbone, "norm"):
            blocks.append(("norm", backbone.norm))
        return blocks

    seen = set()
    blocks = []
    for name, mod in backbone.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) and id(mod) not in seen:
            blocks.append((name, mod))
            seen.add(id(mod))
    return blocks

# Select the last fraction of backbone blocks.
def _select_suffix_blocks(blocks, frac: float):
    if frac <= 0.0:
        return []
    sizes = [(name, mod, _count_params(mod)) for name, mod in blocks]
    total = sum(p for _, _, p in sizes)
    if total <= 0:
        return []
    target = total * frac
    chosen = []
    running = 0
    for name, mod, pcount in reversed(sizes):
        if pcount <= 0:
            continue
        chosen.append((name, mod))
        running += pcount
        if running >= target:
            break
    return chosen

# Measure explanation stability under weight randomization.
def sanity_similarity_curve(model, make_attr_fn, device):
    """
    make_attr_fn(model) -> saliency [1,1,H,W] for a fixed input
    """
    model = model.to(device).eval()

    # baseline attribution
    sal0 = make_attr_fn(model)
    sal0 = normalize_map(sal0)

    # copy model for randomization
    m = copy.deepcopy(model).to(device).eval()
    blocks = _ordered_backbone_blocks(m)
    head = _get_head_module(m)

    out = [{"stage": "original", "sim": 1.0}]
    randomized = set()

    # Stage 1: head-only
    if head is not None:
        randomize_module_weights_(head)
        randomized.add(id(head))
        sal = normalize_map(make_attr_fn(m))
        out.append({"stage": "rand_head", "sim": pearson_corr_flat(sal0, sal)})

    # Progressive backbone randomization: last 25%, 50%, 100%
    for frac, label in [(0.25, "rand_last25"), (0.50, "rand_last50"), (1.00, "rand_all")]:
        for _, blk in _select_suffix_blocks(blocks, frac):
            if id(blk) in randomized:
                continue
            randomize_module_weights_(blk)
            randomized.add(id(blk))
        sal = normalize_map(make_attr_fn(m))
        out.append({"stage": label, "sim": pearson_corr_flat(sal0, sal)})

    return out
