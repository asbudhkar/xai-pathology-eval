#!/usr/bin/env python3
import argparse
import json
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn import make_uni, make_virchow
from src.datasets.pathmnist import get_pathmnist_loaders
from src.datasets.bloodmnist import get_bloodmnist_loaders
from src.datasets.pcam import get_pcam_loaders
from src.models.vit_masking import build_token_keep_mask


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if name == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {name}")


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    num_classes = int(ckpt.get("num_classes", 2))

    model_name = args.get("model", None)
    if model_name == "uni":
        model = make_uni(num_classes, pretrained=False, freeze_backbone=False)
    elif model_name == "virchow":
        model = make_virchow(num_classes, pretrained=False, freeze_backbone=False)
    else:
        raise ValueError(f"Unsupported model in ckpt args: {model_name}")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, args, num_classes


# Unfreeze the last transformer blocks for fine-tuning.
def set_trainable_blocks(backbone: nn.Module, n_last_blocks: int):
    # Freeze all
    for p in backbone.parameters():
        p.requires_grad = False
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return
    if n_last_blocks <= 0:
        return
    for blk in blocks[-n_last_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True


# Data loaders for different datasets
def build_loaders(ds_name: str, img_size: int, batch_size: int, num_workers: int, pcam_root: str, pcam_download: bool):
    if ds_name == "pathmnist":
        return get_pathmnist_loaders(img_size=img_size, batch_size=batch_size, num_workers=num_workers)
    if ds_name == "bloodmnist":
        return get_bloodmnist_loaders(img_size=img_size, batch_size=batch_size, num_workers=num_workers)
    if ds_name == "pcam":
        return get_pcam_loaders(root=pcam_root, img_size=img_size, batch_size=batch_size,
                                num_workers=num_workers, download=pcam_download)
    raise ValueError(f"Unknown dataset: {ds_name}")


# Compute batch accuracy.
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset", required=True, choices=["bloodmnist", "pathmnist", "pcam"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_blocks", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--unfreeze_blocks", type=int, default=2)
    ap.add_argument("--mask_ratio_min", type=float, default=0.0)
    ap.add_argument("--mask_ratio_max", type=float, default=0.7)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcam_root", default="./pcam")
    ap.add_argument("--pcam_download", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)

    model, train_args, num_classes = load_model_from_ckpt(args.ckpt, device)
    model_name = train_args.get("model", "uni")

    # Unfreeze head + last N blocks
    set_trainable_blocks(model.backbone, args.unfreeze_blocks)
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Optimizer with param groups
    params_head = [p for p in model.classifier.parameters() if p.requires_grad]
    params_blocks = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": params_head, "lr": args.lr_head},
         {"params": params_blocks, "lr": args.lr_blocks}],
        weight_decay=args.weight_decay,
    )
    ce = nn.CrossEntropyLoss()

    train_dl, val_dl, _, _ = build_loaders(args.dataset, args.img_size, args.batch_size,
                                           args.num_workers, args.pcam_root, args.pcam_download)

    best_acc = -1.0
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_seen = 0

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.view(-1).long().to(device, non_blocking=True)

            spec = build_token_keep_mask(x, model.backbone, args.mask_ratio_min, args.mask_ratio_max)

            opt.zero_grad(set_to_none=True)
            logits = model.forward_masked(x, token_keep_mask=spec.token_keep_mask)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n_seen += bs

        train_loss = total_loss / max(1, n_seen)
        train_acc = total_acc / max(1, n_seen)

        # eval (unmasked)
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_seen = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                y = y.view(-1).long().to(device, non_blocking=True)
                logits = model(x)
                loss = ce(logits, y)
                bs = x.size(0)
                val_loss += loss.item() * bs
                val_acc += accuracy(logits, y) * bs
                n_seen += bs
        val_loss /= max(1, n_seen)
        val_acc /= max(1, n_seen)

        print(f"ep {ep} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "args": {
                    **deepcopy(train_args),
                    "dataset": args.dataset,
                    "model": model_name,
                    "img_size": args.img_size,
                    "mask_ratio_min": args.mask_ratio_min,
                    "mask_ratio_max": args.mask_ratio_max,
                    "mask_finetune": True,
                },
            }
            torch.save(ckpt, os.path.join(args.outdir, "ckpt_best.pt"))

    meta = {
        "elapsed_sec": time.time() - t0,
        "best_val_acc": best_acc,
        "mask_ratio_min": args.mask_ratio_min,
        "mask_ratio_max": args.mask_ratio_max,
        "unfreeze_blocks": args.unfreeze_blocks,
    }
    with open(os.path.join(args.outdir, "masking_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
