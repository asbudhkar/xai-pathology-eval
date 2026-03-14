import os, json, argparse, random, time, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd

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
from src.eval.sanity import sanity_similarity_curve


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
    args = ckpt["args"]
    num_classes = ckpt["num_classes"]

    if args["model"] == "resnet18":
        model = make_resnet18(num_classes, pretrained=False)
    elif args["model"] == "efficientnet_b0":
        model = make_efficientnet_b0(num_classes, pretrained=False)
    elif args["model"] == "uni":
        model = make_uni(num_classes, pretrained=False, freeze_backbone=True)
    elif args["model"] == "virchow":
        model = make_virchow(num_classes, pretrained=False, freeze_backbone=True)
    else:
        raise ValueError(f"Unsupported model in ckpt: {args['model']}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args, num_classes


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


def main():
    ap = argparse.ArgumentParser()
    PAPER_SALIENCY_MODE = "magnitude"
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", default="./outputs/results/pathmnist_resnet18_seed0")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--baseline", default="mean", choices=["mean","zero","blur","blur21"])
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_mode", default="pred", choices=["true","pred"])
    ap.add_argument("--correct_only", action="store_true",
                    help="Use only correctly predicted samples (pred == y_true)")
    ap.add_argument("--explainer", default="ig",
                    choices=["ig","gradcam","attnrollout","attnrollout_plain","vitshapley"])
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
    ap.add_argument("--saliency_mode", default="magnitude",
                    choices=["magnitude", "positive", "negative"])
    ap.add_argument("--paper_mode", action="store_true",
                    help="Enforce paper settings (requires --indices_in and fixed saliency_mode)")
    ap.add_argument("--score_type", default="prob", choices=["prob","logit","margin"])
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
        idxs = list(range(len(test_ds)))
        chosen = random.sample(idxs, min(args.n, len(idxs)))
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

    if indices_loaded and len(chosen) > args.n:
        chosen = chosen[:args.n]

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

    all_rows = []
    subset_loader = make_subset_loader(test_ds, chosen, args.num_workers)
    for j, (x, y) in enumerate(subset_loader):
        idx = chosen[j]
        x = x.to(device, non_blocking=True)
        y = int(y.squeeze().item())

        with torch.no_grad():
            pred = int(model(x).argmax(dim=1).item())
        target = y if args.target_mode == "true" else pred

        # select explainer
        def make_attr_fn(mdl):
            if args.explainer == "ig":
                attr = ig_attribution(mdl, x, target, baseline=args.baseline, n_steps=args.ig_steps)
            elif args.explainer == "gradcam":
                attr = gradcam_attribution(mdl, x, target)
            elif args.explainer in ["attnrollout", "attn_rollout", "attn_grad"]:
                attr = attention_rollout_attribution(mdl, x, target, use_grad=True)
            elif args.explainer in ["attnrollout_plain", "attn_rollout_plain"]:
                attr = attention_rollout_attribution(mdl, x, target, use_grad=False)
            elif args.explainer == "vitshapley":
                cfg = ViTShapleyConfig(
                    score_mode=args.vs_score_mode,
                    normalize=bool(args.vs_normalize),
                )
                attr = vitshapley_attribution(mdl, x, explainer, n_prefix, target=target, cfg=cfg)
            else:
                raise ValueError(f"Unsupported explainer: {args.explainer}")

            sal_signed = reduce_attribution(attr)
            return saliency_for_ranking(sal_signed, mode=args.saliency_mode)

        curve = sanity_similarity_curve(
            model=model,
            make_attr_fn=make_attr_fn,
            device=device,
        )

        for t, row in enumerate(curve):
            all_rows.append({
                "img_i": j,
                "ds_idx": idx,
                "true": y,
                "pred": pred,
                "target": target,
                "stage_step": t,
                "stage": row["stage"],
                "sim": row["sim"],
            })

        if (j+1) % 10 == 0:
            print(f"[{j+1}/{len(chosen)}] done")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.outdir, f"sanity_weights_{args.explainer}.csv")
    df.to_csv(csv_path, index=False)

    summ = df.groupby("stage")["sim"].agg(["mean","std","count"]).reset_index()
    summary_path = os.path.join(args.outdir, f"sanity_weights_{args.explainer}_summary.csv")
    summ.to_csv(summary_path, index=False)

    t_total = time.perf_counter() - t_start
    meta = {
        "n_images": int(len(chosen)),
        "explainer": args.explainer,
        "target_mode": args.target_mode,
        "score_type": args.score_type,
        "saliency_mode": args.saliency_mode,
        "time_total_sec": float(t_total),
        "time_per_image_sec": float(t_total / max(1, len(chosen))),
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
    meta_path = os.path.join(args.outdir, f"sanity_weights_{args.explainer}_summary.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.outdir, f"run_config_{args.explainer}.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("Saved:", meta_path)

if __name__ == "__main__":
    main()
