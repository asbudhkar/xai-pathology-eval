import os, json, argparse, random, time, hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.models.cnn import make_resnet18, make_efficientnet_b0, make_uni, make_virchow
from src.datasets.pathmnist import get_pathmnist_loaders
from src.datasets.bloodmnist import get_bloodmnist_loaders
from src.datasets.pcam import get_pcam_loaders

from src.explain.ig import ig_attribution
from src.explain.gradcam import gradcam_attribution
from src.explain.attn_rollout import attention_rollout_attribution
from src.explain.vitshapley import vitshapley_attribution, ViTShapleyConfig, ViTShapleyExplainer
from src.models.vit_masking import num_prefix_tokens

VITSHAPLEY_EXPLAINER = None
VITSHAPLEY_PREFIX = None
from src.explain.utils import reduce_attribution, saliency_for_ranking

from src.eval.robustness import hflip, vflip, unflip_map, make_jitter, stability_one


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
    num_classes = ckpt.get("num_classes", None)

    if args.get("model", None) == "resnet18":
        model = make_resnet18(num_classes, pretrained=False)
    elif args.get("model", None) == "efficientnet_b0":
        model = make_efficientnet_b0(num_classes, pretrained=False)
    elif args.get("model", None) == "uni":
        model = make_uni(num_classes, pretrained=False, freeze_backbone=True)
    elif args.get("model", None) == "virchow":
        model = make_virchow(num_classes, pretrained=False, freeze_backbone=True)
    else:
        raise ValueError(f"Unsupported model in ckpt args: {args.get('model', None)}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args, num_classes


# Group dataset indices by class label.
def stratified_indices(ds, num_classes):
    """
    Returns list-of-lists buckets. Auto-expands if a label id exceeds num_classes.
    This prevents crashes when ckpt num_classes is wrong (e.g., PCAM mistakenly saved as 1).
    """
    if num_classes is None:
        buckets = []
    else:
        buckets = [[] for _ in range(int(num_classes))]

    for i in range(len(ds)):
        _, y = ds[i]
        y = int(np.array(y).squeeze())
        if y < 0:
            continue

        if y >= len(buckets):
            buckets.extend([[] for _ in range(y - len(buckets) + 1)])
        buckets[y].append(i)

    return buckets


def make_subset_loader(ds, indices, num_workers: int):
    subset = Subset(ds, indices)
    dl_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return DataLoader(subset, **{k: v for k, v in dl_kwargs.items() if v is not None})


def _load_indices(path: str) -> list[int]:
    df = pd.read_csv(path)
    if "idx" not in df.columns:
        raise ValueError(f"Missing 'idx' column in indices file: {path}")
    return df["idx"].astype(int).tolist()


def _save_indices(path: str, indices: list[int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame({"idx": [int(v) for v in indices]})
    df.to_csv(path, index=False)


# Compute one attribution map for the chosen explainer.
def compute_attr(model, x, target, args):
    """
    Returns attribution tensor (shape depends on explainer; reduce_attribution handles it).
    x: [1,C,H,W] on device
    """
    ex = args.explainer

    if ex == "ig":
        return ig_attribution(
            model, x, target,
            baseline=args.baseline,
            n_steps=int(args.ig_steps),
        )

    if ex == "gradcam":
        return gradcam_attribution(model, x, target)

    if ex in ["attnrollout", "attn_rollout", "attn_grad"]:
        return attention_rollout_attribution(model, x, target, use_grad=True)

    if ex in ["attnrollout_plain", "attn_rollout_plain"]:
        return attention_rollout_attribution(model, x, target, use_grad=False)


    if ex == "vitshapley":
        cfg = ViTShapleyConfig(
            score_mode=args.vs_score_mode,
            normalize=bool(args.vs_normalize),
        )
        return vitshapley_attribution(model, x, VITSHAPLEY_EXPLAINER, VITSHAPLEY_PREFIX,
                                      target=target, cfg=cfg)

    raise ValueError(f"Unknown explainer: {ex}")


def main():
    ap = argparse.ArgumentParser()
    PAPER_SALIENCY_MODE = "magnitude"
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", default="./outputs/results/pathmnist_resnet18_seed0")
    ap.add_argument("--n", type=int, default=300)

    ap.add_argument("--baseline", default="mean", choices=["mean", "zero", "blur", "blur21"])
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_mode", default="pred", choices=["true", "pred"])
    ap.add_argument("--correct_only", action="store_true",
                    help="Use only correctly predicted samples (pred == y_true)")

    ap.add_argument("--saliency_mode", default="magnitude",
                    choices=["magnitude", "positive", "negative"])
    ap.add_argument("--paper_mode", action="store_true",
                    help="Enforce paper settings (requires --indices_in and fixed saliency_mode)")
    ap.add_argument("--score_type", default="prob", choices=["prob", "logit", "margin"])
    ap.add_argument("--indices_in", default=None, type=str,
                    help="Path to JSON list of dataset indices to evaluate")
    ap.add_argument("--indices_out", default=None, type=str,
                    help="Path to save JSON list of chosen indices")
    ap.add_argument("--pcam_root", default="./pcam",
                    help="PCAM root folder (PCAM will create/use <root>/pcam)")
    ap.add_argument("--pcam_download", action="store_true",
                    help="Download PCAM if missing (uses --pcam_root)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="Device selection (auto prefers CUDA then MPS)")

    ap.add_argument("--explainer", default="ig",
                    choices=["ig", "gradcam", "attnrollout", "attnrollout_plain", "vitshapley"])

    # SmoothGrad
    ap.add_argument("--sg_samples", type=int, default=12)
    ap.add_argument("--sg_sigma", type=float, default=0.10)

    # ViTShapley
    ap.add_argument("--vs_grid", type=int, default=14)
    ap.add_argument("--vs_perms", type=int, default=32)
    ap.add_argument("--vs_step", type=int, default=1)
    ap.add_argument("--vs_baseline", default="mean", choices=["mean", "zero", "blur"])
    ap.add_argument("--vs_score_mode", default="logit", choices=["logit", "prob"])
    ap.add_argument("--vs_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vs_explainer", default=None, help="Path to ViT-Shapley explainer weights")

    args = ap.parse_args()
    if args.paper_mode:
        if not args.indices_in:
            ap.error("--paper_mode requires --indices_in for reproducible indices.")
        if args.saliency_mode != PAPER_SALIENCY_MODE:
            ap.error(f"--paper_mode requires --saliency_mode={PAPER_SALIENCY_MODE}.")
    print(f"[RUN] indices_in={args.indices_in} saliency_mode={args.saliency_mode} target_mode={args.target_mode}")

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    model, train_args, num_classes = load_model_from_ckpt(args.ckpt, device)
    global VITSHAPLEY_EXPLAINER, VITSHAPLEY_PREFIX
    VITSHAPLEY_EXPLAINER = None
    VITSHAPLEY_PREFIX = None
    if args.explainer == "vitshapley":
        if not args.vs_explainer:
            raise RuntimeError("--vs_explainer is required for vitshapley.")
        embed_dim = getattr(model.backbone, "num_features", None)
        if embed_dim is None:
            raise RuntimeError("Backbone missing num_features; cannot build ViTShapley explainer.")
        n_classes = int(num_classes) if num_classes is not None else int(getattr(model.classifier, "out_features", 1))
        VITSHAPLEY_EXPLAINER = ViTShapleyExplainer(embed_dim=embed_dim, num_classes=n_classes)
        VITSHAPLEY_EXPLAINER.load_state_dict(torch.load(args.vs_explainer, map_location="cpu"))
        VITSHAPLEY_EXPLAINER.to(device).eval()
        VITSHAPLEY_PREFIX = num_prefix_tokens(model.backbone)
    ds_name = str(train_args.get("dataset", "pathmnist")).lower()
    img_size = int(train_args.get("img_size", 224))

    if ds_name == "pathmnist":
        _, _, test_dl, _ = get_pathmnist_loaders(img_size, batch_size=1, num_workers=0)
    elif ds_name == "bloodmnist":
        _, _, test_dl, _ = get_bloodmnist_loaders(img_size, batch_size=1, num_workers=0)
    elif ds_name == "pcam":
        _, _, test_dl, _ = get_pcam_loaders(
            root=args.pcam_root,
            img_size=img_size,
            batch_size=1,
            num_workers=0,
            download=bool(args.pcam_download),
        )
    else:
        raise ValueError(f"Unknown dataset in ckpt args: {ds_name}")

    test_ds = test_dl.dataset
    default_indices_path = os.path.join(args.outdir, f"eval_indices_{ds_name}_normal.csv")
    indices_path = None
    indices_loaded = False
    if args.indices_in:
        indices_path = args.indices_in
    elif os.path.exists(default_indices_path):
        indices_path = default_indices_path

    if indices_path:
        chosen = _load_indices(indices_path)
        indices_loaded = True
    else:
        buckets = stratified_indices(test_ds, num_classes)

        # Determine number of classes robustly
        if num_classes is None:
            n_classes = len(buckets) if len(buckets) > 0 else 1
        else:
            n_classes = max(int(num_classes), len(buckets)) if len(buckets) > 0 else int(num_classes)

        # Stratified sampling
        per_class = max(1, args.n // max(1, n_classes))
        chosen = []

        for c in range(n_classes):
            if c >= len(buckets):
                continue
            idxs = buckets[c]
            if not idxs:
                continue
            take = min(per_class, len(idxs))
            chosen.extend(random.sample(idxs, take))

        if len(chosen) < args.n:
            remaining = list(set(range(len(test_ds))) - set(chosen))
            if remaining:
                chosen.extend(random.sample(remaining, min(args.n - len(chosen), len(remaining))))

        chosen = chosen[:args.n]
        random.shuffle(chosen)
        indices_path = args.indices_out or default_indices_path

    # Handle the is correct index step.
    def is_correct_index(idx):
        x, y = test_ds[idx]
        if torch.is_tensor(y):
            y = int(y.squeeze().item())
        elif isinstance(y, np.ndarray):
            y = int(y.reshape(-1)[0])
        else:
            y = int(y)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            pred = int(model(x).argmax(dim=1).item())
        return pred == y

    if args.correct_only and not indices_loaded:
        correct = [idx for idx in chosen if is_correct_index(idx)]
        if len(correct) < args.n:
            remaining = list(set(range(len(test_ds))) - set(correct))
            random.shuffle(remaining)
            for idx in remaining:
                if len(correct) >= args.n:
                    break
                if is_correct_index(idx):
                    correct.append(idx)
        chosen = correct

    if not indices_loaded:
        _save_indices(indices_path, chosen)

    hash8 = hashlib.sha1(",".join(str(i) for i in chosen).encode("utf-8")).hexdigest()[:8]
    print(f"[INDICES] n={len(chosen)} path={indices_path} sha1={hash8}")

    jitter = make_jitter()
    rows = []

    subset_loader = make_subset_loader(test_ds, chosen, args.num_workers)
    for j, (x, y) in enumerate(subset_loader):
        idx = chosen[j]
        x_cpu = x[0]  # [C,H,W] on CPU for jitter
        x = x.to(device, non_blocking=True)
        y = int(y.squeeze().item())

        with torch.no_grad():
            pred = int(model(x).argmax(dim=1).item())
        target = y if args.target_mode == "true" else pred

        # Original
        attr0 = compute_attr(model, x, target, args)
        sal0_signed = reduce_attribution(attr0)  # [1,1,H,W]
        sal0 = saliency_for_ranking(sal0_signed, mode=args.saliency_mode)

        # H flip
        xh = hflip(x)
        with torch.no_grad():
            pred_h = int(model(xh).argmax(dim=1).item())
        attr_h = compute_attr(model, xh, target, args)
        sal_h_signed = reduce_attribution(attr_h)
        sal_h = saliency_for_ranking(sal_h_signed, mode=args.saliency_mode)
        sal_h = unflip_map(sal_h, "h")

        # V flip
        xv = vflip(x)
        with torch.no_grad():
            pred_v = int(model(xv).argmax(dim=1).item())
        attr_v = compute_attr(model, xv, target, args)
        sal_v_signed = reduce_attribution(attr_v)
        sal_v = saliency_for_ranking(sal_v_signed, mode=args.saliency_mode)
        sal_v = unflip_map(sal_v, "v")

        # Color jitter
        xj = jitter(x_cpu).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            pred_j = int(model(xj).argmax(dim=1).item())
        attr_j = compute_attr(model, xj, target, args)
        sal_j_signed = reduce_attribution(attr_j)
        sal_j = saliency_for_ranking(sal_j_signed, mode=args.saliency_mode)

        s_h = stability_one(sal0, sal_h)
        s_v = stability_one(sal0, sal_v)
        s_j = stability_one(sal0, sal_j)

        same_h = (pred_h == pred)
        same_v = (pred_v == pred)
        same_j = (pred_j == pred)

        rows.append({
            "i": j,
            "idx": int(idx),
            "true": int(y),
            "pred": int(pred),
            "target": int(target),
            "stab_hflip": float(s_h),
            "stab_vflip": float(s_v),
            "stab_jitter": float(s_j),
            "samepred_hflip": int(same_h),
            "samepred_vflip": int(same_v),
            "samepred_jitter": int(same_j),
            "stab_hflip_samepred": float(s_h) if same_h else np.nan,
            "stab_vflip_samepred": float(s_v) if same_v else np.nan,
            "stab_jitter_samepred": float(s_j) if same_j else np.nan,
        })

        if (j + 1) % 25 == 0:
            arr = np.array([r["stab_hflip"] for r in rows], dtype=float)
            print(f"[{j+1}/{len(chosen)}] mean hflip stab={arr.mean():.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, f"robustness_{args.explainer}.csv")
    df.to_csv(csv_path, index=False)
    t_total = time.perf_counter() - t_start

    summary = {
        "n": int(len(df)),
        "baseline": args.baseline,
        "ig_steps": int(args.ig_steps),
        "target_mode": args.target_mode,
        "score_type": args.score_type,
        "saliency_mode": args.saliency_mode,
        "explainer": args.explainer,
        "hflip_mean": float(df["stab_hflip"].mean()),
        "hflip_std": float(df["stab_hflip"].std(ddof=1)),
        "vflip_mean": float(df["stab_vflip"].mean()),
        "vflip_std": float(df["stab_vflip"].std(ddof=1)),
        "jitter_mean": float(df["stab_jitter"].mean()),
        "jitter_std": float(df["stab_jitter"].std(ddof=1)),
        "hflip_samepred_mean": float(df["stab_hflip_samepred"].mean()),
        "hflip_samepred_count": int(df["stab_hflip_samepred"].notna().sum()),
        "vflip_samepred_mean": float(df["stab_vflip_samepred"].mean()),
        "vflip_samepred_count": int(df["stab_vflip_samepred"].notna().sum()),
        "jitter_samepred_mean": float(df["stab_jitter_samepred"].mean()),
        "jitter_samepred_count": int(df["stab_jitter_samepred"].notna().sum()),
        "time_total_sec": float(t_total),
        "time_per_sample_sec": float(t_total / max(1, len(df))),
    }
    run_config = {
        "dataset": train_args.get("dataset", None),
        "model": train_args.get("model", None),
        "seed": int(args.seed),
        "n": int(args.n),
        "indices_path": indices_path,
        "target_mode": args.target_mode,
        "correct_only": bool(args.correct_only),
        "explainer": args.explainer,
        "score_type": args.score_type,
        "saliency_mode": args.saliency_mode,
        "fill": None,
        "baseline": args.baseline,
        "step_schedule": None,
    }
    with open(os.path.join(args.outdir, f"robustness_{args.explainer}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, f"run_config_{args.explainer}.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print("Saved:", csv_path)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
