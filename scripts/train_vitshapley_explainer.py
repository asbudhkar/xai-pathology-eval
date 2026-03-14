#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch

from src.models.cnn import make_uni, make_virchow
from src.datasets.pathmnist import get_pathmnist_loaders
from src.datasets.bloodmnist import get_bloodmnist_loaders
from src.datasets.pcam import get_pcam_loaders
from src.models.vit_masking import (
    build_token_keep_mask,
    build_token_keep_mask_uniform_cardinality,
    num_prefix_tokens,
)
from src.explain.vitshapley import ViTShapleyExplainer, vitshapley_loss


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
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, args, num_classes


def build_loaders(ds_name: str, img_size: int, batch_size: int, num_workers: int, pcam_root: str, pcam_download: bool):
    if ds_name == "pathmnist":
        return get_pathmnist_loaders(img_size=img_size, batch_size=batch_size, num_workers=num_workers)
    if ds_name == "bloodmnist":
        return get_bloodmnist_loaders(img_size=img_size, batch_size=batch_size, num_workers=num_workers)
    if ds_name == "pcam":
        return get_pcam_loaders(root=pcam_root, img_size=img_size, batch_size=batch_size,
                                num_workers=num_workers, download=pcam_download)
    raise ValueError(f"Unknown dataset: {ds_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset", required=True, choices=["bloodmnist", "pathmnist", "pcam"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--mask_ratio_min", type=float, default=0.0)
    ap.add_argument("--mask_ratio_max", type=float, default=0.8)
    ap.add_argument("--mask_sampling", default="uniform_cardinality",
                    choices=["uniform_cardinality", "uniform_ratio"])
    ap.add_argument("--score_mode", default="logit", choices=["logit", "prob"])
    ap.add_argument("--l2_lambda", type=float, default=0.0)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_mode", default="pred", choices=["pred", "true"])
    ap.add_argument("--pcam_root", default="./pcam")
    ap.add_argument("--pcam_download", action="store_true")
    ap.add_argument("--surrogate_ckpt", default=None,
                    help="Optional surrogate model ckpt for masked inputs (paper-mode).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)

    model, train_args, num_classes = load_model_from_ckpt(args.ckpt, device)
    n_prefix = num_prefix_tokens(model.backbone)
    score_model = model
    if args.surrogate_ckpt:
        score_model, _, _ = load_model_from_ckpt(args.surrogate_ckpt, device)

    # Build explainer
    embed_dim = getattr(model.backbone, "num_features", None)
    if embed_dim is None:
        raise ValueError("Backbone missing num_features; cannot build explainer.")
    explainer = ViTShapleyExplainer(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=int(num_classes) if num_classes is not None else 1,
    )
    explainer.to(device).train()

    opt = torch.optim.AdamW(explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dl, _, _, _ = build_loaders(args.dataset, args.img_size, args.batch_size,
                                      args.num_workers, args.pcam_root, args.pcam_download)

    os.makedirs(args.outdir, exist_ok=True)
    best_loss = float("inf")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        explainer.train()
        total_loss = 0.0
        n_seen = 0

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.view(-1).long().to(device, non_blocking=True)

            with torch.no_grad():
                logits_full = model(x)
                if args.target_mode == "pred":
                    target = logits_full.argmax(dim=1)
                else:
                    target = y

            if args.mask_sampling == "uniform_ratio":
                spec = build_token_keep_mask(x, model.backbone, args.mask_ratio_min, args.mask_ratio_max)
            else:
                spec = build_token_keep_mask_uniform_cardinality(x, model.backbone)

            opt.zero_grad(set_to_none=True)
            loss = vitshapley_loss(
                model=model,
                explainer=explainer,
                x=x,
                target=target,
                token_keep_mask=spec.token_keep_mask,
                num_prefix_tokens=n_prefix,
                score_mode=args.score_mode,
                score_model=score_model,
                l2_lambda=args.l2_lambda,
            )
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

        mean_loss = total_loss / max(1, n_seen)
        print(f"ep {ep} | loss {mean_loss:.6f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(explainer.state_dict(), os.path.join(args.outdir, "explainer_best.pt"))

    meta = {
        "elapsed_sec": time.time() - t0,
        "best_loss": best_loss,
        "mask_sampling": args.mask_sampling,
        "mask_ratio_min": args.mask_ratio_min,
        "mask_ratio_max": args.mask_ratio_max,
        "score_mode": args.score_mode,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "surrogate_ckpt": args.surrogate_ckpt,
    }
    with open(os.path.join(args.outdir, "train_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
