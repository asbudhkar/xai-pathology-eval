#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from src.models.cnn import make_uni, make_virchow
from src.datasets.pathmnist import get_pathmnist_loaders
from src.datasets.bloodmnist import get_bloodmnist_loaders
from src.datasets.pcam import get_pcam_loaders
from src.models.vit_masking import (
    apply_patch_mask,
    build_token_keep_mask,
    build_token_keep_mask_uniform_cardinality,
)


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


def load_model_from_ckpt(ckpt_path: str, device: torch.device, trainable: bool):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    num_classes = int(ckpt.get("num_classes", 2))

    model_name = args.get("model", None)
    if model_name == "uni":
        model = make_uni(num_classes, pretrained=False, freeze_backbone=not trainable)
    elif model_name == "virchow":
        model = make_virchow(num_classes, pretrained=False, freeze_backbone=not trainable)
    else:
        raise ValueError(f"Unsupported model in ckpt args: {model_name}")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    if not trainable:
        model.eval()
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


# Run the script entrypoint.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Classifier ckpt to approximate")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset", required=True, choices=["bloodmnist", "pathmnist", "pcam"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--mask_sampling", default="uniform_cardinality",
                    choices=["uniform_cardinality", "uniform_ratio"])
    ap.add_argument("--mask_ratio_min", type=float, default=0.0)
    ap.add_argument("--mask_ratio_max", type=float, default=0.8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcam_root", default="./pcam")
    ap.add_argument("--pcam_download", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)

    # f: fixed classifier, g: trainable surrogate
    model_f, train_args, _ = load_model_from_ckpt(args.ckpt, device, trainable=False)
    model_g, _, _ = load_model_from_ckpt(args.ckpt, device, trainable=True)

    opt = torch.optim.AdamW(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dl, _, _, _ = build_loaders(args.dataset, args.img_size, args.batch_size,
                                      args.num_workers, args.pcam_root, args.pcam_download)

    os.makedirs(args.outdir, exist_ok=True)
    best_loss = float("inf")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model_g.train()
        total_loss = 0.0
        n_seen = 0

        for x, _ in train_dl:
            x = x.to(device, non_blocking=True)

            if args.mask_sampling == "uniform_ratio":
                spec = build_token_keep_mask(x, model_g.backbone, args.mask_ratio_min, args.mask_ratio_max)
            else:
                spec = build_token_keep_mask_uniform_cardinality(x, model_g.backbone)

            x_masked = apply_patch_mask(x, model_g.backbone, spec.token_keep_mask, fill=0.0)

            with torch.no_grad():
                logits_f = model_f(x)
                prob_f = torch.softmax(logits_f, dim=1)

            opt.zero_grad(set_to_none=True)
            logits_g = model_g(x_masked)
            logprob_g = torch.log_softmax(logits_g, dim=1)
            loss = F.kl_div(logprob_g, prob_f, reduction="batchmean")
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

        mean_loss = total_loss / max(1, n_seen)
        print(f"ep {ep} | kl {mean_loss:.6f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            ckpt = {
                "model": model_g.state_dict(),
                "num_classes": getattr(model_g.classifier, "out_features", None),
                "args": train_args,
            }
            torch.save(ckpt, os.path.join(args.outdir, "surrogate_best.pt"))

    meta = {
        "elapsed_sec": time.time() - t0,
        "best_kl": best_loss,
        "mask_sampling": args.mask_sampling,
        "mask_ratio_min": args.mask_ratio_min,
        "mask_ratio_max": args.mask_ratio_max,
    }
    with open(os.path.join(args.outdir, "surrogate_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
