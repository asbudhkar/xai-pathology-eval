import os, json, argparse, random, time, hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from src.models.cnn import make_resnet18, make_efficientnet_b0, make_uni, make_virchow
from src.datasets.pathmnist import get_pathmnist_loaders
from src.datasets.bloodmnist import get_bloodmnist_loaders
from src.datasets.pcam import get_pcam_loaders

from src.explain.ig import ig_attribution
from src.explain.gradcam import gradcam_attribution
from src.explain.attn_rollout import attention_rollout_attribution
from src.explain.vitshapley import vitshapley_attribution, ViTShapleyConfig, ViTShapleyExplainer
from src.models.vit_masking import num_prefix_tokens

from src.explain.utils import reduce_attribution, saliency_for_ranking
from src.eval.faithfulness import insertion_deletion_acc_auc, insertion_deletion_auc, mask_fill_image


# Compute macro one-vs-rest AUROC.
def _safe_macro_ovr_auroc(y_true: np.ndarray, prob: np.ndarray):
    try:
        return float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except ValueError:
        return None


# Compute AUROC for saved predictions.
def _compute_auroc(y_true: list[int], probs: list[np.ndarray]):
    if not probs:
        return None
    y = np.array(y_true, dtype=int)
    P = np.stack(probs, axis=0)
    if P.ndim == 1:
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, P))
    if P.shape[1] == 2:
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, P[:, 1]))
    return _safe_macro_ovr_auroc(y, P)


# Choose device
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


# Load a saved model checkpoint and build the model.
def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    num_classes = int(ckpt["num_classes"])

    model_name = train_args.get("model", None) if isinstance(train_args, dict) else getattr(train_args, "model", None)
    if model_name == "resnet18":
        model = make_resnet18(num_classes, pretrained=False)
    elif model_name == "efficientnet_b0":
        model = make_efficientnet_b0(num_classes, pretrained=False)
    elif model_name == "uni":
        model = make_uni(num_classes, pretrained=False, freeze_backbone=True)
    elif model_name == "virchow":
        model = make_virchow(num_classes, pretrained=False, freeze_backbone=True)
    else:
        raise ValueError(f"Unsupported model in ckpt args: {model_name}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, train_args, num_classes


def aget(a, k, default=None):
    """Get attribute or dict key."""
    if isinstance(a, dict):
        return a.get(k, default)
    return getattr(a, k, default)


# Group dataset indices by class label.
def stratified_indices_from_dataset(ds, num_classes):
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


# Build a dataloader for a selected subset of samples.
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


# Compute the channel mean over selected samples.
def dataset_channel_mean(ds, indices: list[int]) -> torch.Tensor:
    """Compute per-channel mean over selected indices (in [0,1] pixel space)."""
    total = None
    count = 0
    for idx in indices:
        x, _ = ds[int(idx)]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.float()
        if total is None:
            total = x.sum(dim=(-2, -1))
        else:
            total += x.sum(dim=(-2, -1))
        count += int(x.shape[-2] * x.shape[-1])
    if total is None or count == 0:
        return torch.tensor(0.0)
    return total / float(count)


def main():
    ap = argparse.ArgumentParser()
    PAPER_SALIENCY_MODE = "magnitude"

    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--outdir", default="./outputs/results/pathmnist_resnet18_seed0", type=str)

    ap.add_argument("--n", default=300, type=int)              # total samples
    ap.add_argument("--out_suffix", default="", type=str,
                    help="Optional suffix for output filenames (e.g., _mean)")

    ap.add_argument("--baseline", default="mean", choices=["mean", "zero", "blur", "blur21"])
    ap.add_argument("--ig_steps", default=32, type=int)

    ap.add_argument("--fill", default="blur", choices=["blur", "mean", "zero"])
    ap.add_argument("--seed", default=0, type=int)
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
    ap.add_argument("--del_pct", type=float, default=0.20,
                    help="Fraction of tiles to delete for top-k vs random deletion ablation")

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

    explainer = None
    n_prefix = None
    if args.explainer == "vitshapley":
        if not args.vs_explainer:
            raise RuntimeError("--vs_explainer is required for vitshapley.")
        embed_dim = getattr(model.backbone, "num_features", None)
        if embed_dim is None:
            raise RuntimeError("Backbone missing num_features; cannot build ViTShapley explainer.")
        n_classes = int(num_classes) if num_classes is not None else int(getattr(model.classifier, "out_features", 1))
        explainer = ViTShapleyExplainer(embed_dim=embed_dim, num_classes=n_classes)
        explainer.load_state_dict(torch.load(args.vs_explainer, map_location="cpu"))
        explainer.to(device).eval()
        n_prefix = num_prefix_tokens(model.backbone)
    ds_name = str(aget(train_args, "dataset", "pathmnist")).lower()
    img_size = int(aget(train_args, "img_size", 224))

    # Load dataset (batch_size=1 so we can directly index dataset)
    if ds_name == "pathmnist":
        _, _, test_dl, _ = get_pathmnist_loaders(img_size=img_size, batch_size=1, num_workers=0)
    elif ds_name == "bloodmnist":
        _, _, test_dl, _ = get_bloodmnist_loaders(img_size=img_size, batch_size=1, num_workers=0)
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
        # Stratified sampling
        buckets = stratified_indices_from_dataset(test_ds, num_classes=num_classes)

        if num_classes is None:
            n_classes = len(buckets) if len(buckets) > 0 else 1
        else:
            n_classes = max(int(num_classes), len(buckets)) if len(buckets) > 0 else int(num_classes)

        per_class = max(1, args.n // max(1, n_classes))

        chosen = []
        for c in range(n_classes):
            if c >= len(buckets):
                continue
            idxs = buckets[c]
            if len(idxs) == 0:
                continue
            take = min(per_class, len(idxs))
            chosen.extend(random.sample(idxs, take))

        if len(chosen) < args.n:
            remaining = list(set(range(len(test_ds))) - set(chosen))
            take = min(args.n - len(chosen), len(remaining))
            if take > 0:
                chosen.extend(random.sample(remaining, take))

        chosen = chosen[:args.n]
        random.shuffle(chosen)
        indices_path = args.indices_out or default_indices_path

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

    mean_tensor = None
    if args.fill == "mean":
        full_indices = list(range(len(test_ds)))
        mean_tensor = dataset_channel_mean(test_ds, full_indices).to(device)

    rows = []
    base_probs = []
    del_top_probs = []
    del_rand_probs = []
    y_true_all = []
    del_acc_curves = []
    ins_acc_curves = []
    rng = np.random.default_rng(args.seed)
    subset_loader = make_subset_loader(test_ds, chosen, args.num_workers)
    for j, (x, y) in enumerate(subset_loader):
        idx = chosen[j]
        x = x.to(device, non_blocking=True)  # [1,C,H,W]
        y = int(y.squeeze().item())

        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        target = y if args.target_mode == "true" else pred
        y_true_all.append(int(y))
        base_probs.append(prob)

        # Attribution
        if args.explainer == "ig":
            attr = ig_attribution(model, x, target, baseline=args.baseline, n_steps=args.ig_steps)

        elif args.explainer == "gradcam":
            attr = gradcam_attribution(model, x, target)

        elif args.explainer in ["attnrollout", "attn_rollout", "attn_grad"]:
            attr = attention_rollout_attribution(model, x, target, use_grad=True)

        elif args.explainer in ["attnrollout_plain", "attn_rollout_plain"]:
            attr = attention_rollout_attribution(model, x, target, use_grad=False)

        elif args.explainer == "vitshapley":
            cfg = ViTShapleyConfig(
                score_mode=args.vs_score_mode,
                normalize=bool(args.vs_normalize),
            )
            attr = vitshapley_attribution(model, x, explainer, n_prefix, target=target, cfg=cfg)

        else:
            raise ValueError(f"Unsupported explainer: {args.explainer}")

        sal_signed = reduce_attribution(attr)  # [1,1,H,W]
        sal_rank = saliency_for_ranking(sal_signed, mode=args.saliency_mode)

        # Deletion: drop top saliency pixels vs random pixels
        sal_flat = sal_rank.reshape(-1)
        n_total = int(sal_flat.numel())
        n_del = max(1, int(np.ceil(float(args.del_pct) * n_total)))
        order = torch.argsort(sal_flat, descending=True)
        top_idx = order[:n_del]
        rand_idx = rng.choice(n_total, size=n_del, replace=False)

        base = mask_fill_image(x, fill=args.fill, mean=mean_tensor)
        mask_top = torch.zeros(n_total, device=x.device)
        mask_top[top_idx] = 1.0
        m_top = mask_top.view(1, 1, x.shape[-2], x.shape[-1])
        x_top = x * (1 - m_top) + base * m_top

        mask_rand = torch.zeros(n_total, device=x.device)
        mask_rand[torch.from_numpy(rand_idx).to(x.device)] = 1.0
        m_rand = mask_rand.view(1, 1, x.shape[-2], x.shape[-1])
        x_rand = x * (1 - m_rand) + base * m_rand

        with torch.no_grad():
            logits_top = model(x_top)
            logits_rand = model(x_rand)
            pred_top = int(logits_top.argmax(dim=1).item())
            pred_rand = int(logits_rand.argmax(dim=1).item())
            prob_top = torch.softmax(logits_top, dim=1).detach().cpu().numpy()[0]
            prob_rand = torch.softmax(logits_rand, dim=1).detach().cpu().numpy()[0]

        del_top_probs.append(prob_top)
        del_rand_probs.append(prob_rand)

        del_auc, _ = insertion_deletion_auc(model, x, target, sal_rank,
                                            mode="deletion", fill=args.fill, mean=mean_tensor,
                                            score_type=args.score_type)
        ins_auc, _ = insertion_deletion_auc(model, x, target, sal_rank,
                                            mode="insertion", fill=args.fill, mean=mean_tensor,
                                            score_type=args.score_type)
        del_acc_auc, _ = insertion_deletion_acc_auc(model, x, y, sal_rank,
                                                    mode="deletion", fill=args.fill,
                                                    mean=mean_tensor)
        ins_acc_auc, _ = insertion_deletion_acc_auc(model, x, y, sal_rank,
                                                    mode="insertion", fill=args.fill,
                                                    mean=mean_tensor)
        _, del_acc_curve = insertion_deletion_acc_auc(model, x, y, sal_rank,
                                                      mode="deletion", fill=args.fill,
                                                      mean=mean_tensor)
        _, ins_acc_curve = insertion_deletion_acc_auc(model, x, y, sal_rank,
                                                      mode="insertion", fill=args.fill,
                                                      mean=mean_tensor)
        del_acc_curves.append(del_acc_curve)
        ins_acc_curves.append(ins_acc_curve)

        rows.append({
            "i": j,
            "idx": idx,
            "true": y,
            "pred": pred,
            "target": target,
            "deletion_auc": float(del_auc),
            "insertion_auc": float(ins_auc),
            "deletion_acc_auc": float(del_acc_auc),
            "insertion_acc_auc": float(ins_acc_auc),
            "del_pct": float(args.del_pct),
            "del_top_pred": int(pred_top),
            "del_rand_pred": int(pred_rand),
            "del_top_correct": int(pred_top == y),
            "del_rand_correct": int(pred_rand == y),
        })

        if (j + 1) % 25 == 0:
            mean_del = float(np.mean([r["deletion_auc"] for r in rows]))
            mean_ins = float(np.mean([r["insertion_auc"] for r in rows]))
            mean_del_acc = float(np.mean([r["deletion_acc_auc"] for r in rows]))
            mean_ins_acc = float(np.mean([r["insertion_acc_auc"] for r in rows]))
            print(
                f"[{j+1}/{len(chosen)}] mean del_auc={mean_del:.4f} mean ins_auc={mean_ins:.4f} "
                f"mean del_acc_auc={mean_del_acc:.4f} mean ins_acc_auc={mean_ins_acc:.4f}"
            )

    df = pd.DataFrame(rows)
    suffix = str(args.out_suffix)
    csv_path = os.path.join(args.outdir, f"faithfulness_{args.explainer}{suffix}.csv")
    df.to_csv(csv_path, index=False)
    t_total = time.perf_counter() - t_start

    summary = {
        "n": int(len(df)),
        "del_pct": float(args.del_pct),
        "baseline": args.baseline,
        "fill": args.fill,
        "target_mode": args.target_mode,
        "score_type": args.score_type,
        "saliency_mode": args.saliency_mode,
        "explainer": args.explainer,
        "deletion_auc_mean": float(df["deletion_auc"].mean()),
        "deletion_auc_std": float(df["deletion_auc"].std()),
        "insertion_auc_mean": float(df["insertion_auc"].mean()),
        "insertion_auc_std": float(df["insertion_auc"].std()),
        "deletion_acc_auc_mean": float(df["deletion_acc_auc"].mean()),
        "deletion_acc_auc_std": float(df["deletion_acc_auc"].std()),
        "insertion_acc_auc_mean": float(df["insertion_acc_auc"].mean()),
        "insertion_acc_auc_std": float(df["insertion_acc_auc"].std()),
        "del_top_acc": float(df["del_top_correct"].mean()),
        "del_rand_acc": float(df["del_rand_correct"].mean()),
    }
    summary["base_acc"] = float((df["pred"] == df["true"]).mean())
    summary["del_acc_drop_top"] = float(summary["base_acc"] - summary["del_top_acc"])
    summary["del_acc_drop_rand"] = float(summary["base_acc"] - summary["del_rand_acc"])
    auroc_base = _compute_auroc(y_true_all, base_probs)
    auroc_del_top = _compute_auroc(y_true_all, del_top_probs)
    auroc_del_rand = _compute_auroc(y_true_all, del_rand_probs)
    summary["auroc_base"] = auroc_base
    summary["auroc_del_top"] = auroc_del_top
    summary["auroc_del_rand"] = auroc_del_rand
    summary["auroc_drop_top"] = (auroc_base - auroc_del_top) if (auroc_base is not None and auroc_del_top is not None) else None
    summary["auroc_drop_rand"] = (auroc_base - auroc_del_rand) if (auroc_base is not None and auroc_del_rand is not None) else None
    summary["time_total_sec"] = float(t_total)
    summary["time_per_sample_sec"] = float(t_total / max(1, len(df)))

    run_config = {
        "dataset": aget(train_args, "dataset", None),
        "model": aget(train_args, "model", None),
        "seed": int(args.seed),
        "n": int(args.n),
        "indices_path": indices_path,
        "target_mode": args.target_mode,
        "correct_only": bool(args.correct_only),
        "explainer": args.explainer,
        "score_type": args.score_type,
        "saliency_mode": args.saliency_mode,
        "fill": args.fill,
        "baseline": args.baseline,
        "step_schedule": {"max_pct": 45, "step": 5},
    }

    with open(os.path.join(args.outdir, f"faithfulness_{args.explainer}{suffix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, f"run_config_{args.explainer}{suffix}.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    if del_acc_curves and ins_acc_curves:
        steps_pct = list(range(0, 46, 5))
        curves = {
            "steps_pct": steps_pct,
            "del_acc_mean": np.mean(np.stack(del_acc_curves, axis=0), axis=0).tolist(),
            "ins_acc_mean": np.mean(np.stack(ins_acc_curves, axis=0), axis=0).tolist(),
        }
        with open(os.path.join(args.outdir, f"faithfulness_{args.explainer}{suffix}_acc_curves.json"), "w") as f:
            json.dump(curves, f, indent=2)

    print("Saved:", csv_path)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
