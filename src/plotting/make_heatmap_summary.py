import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Datasets
from medmnist import BloodMNIST, PathMNIST
from torchvision.datasets import PCAM

# explainers
from src.explain.ig import ig_attribution
from src.explain.gradcam import gradcam_attribution
from src.explain.attn_rollout import attention_rollout_attribution
from src.explain.utils import reduce_attribution, normalize_map
from src.explain.vitshapley import ViTShapleyExplainer, ViTShapleyConfig, vitshapley_attribution
from src.models.vit_masking import num_prefix_tokens

DATASET_LABELS = {
    "bloodmnist": "BloodMNIST",
    "pathmnist": "PathMNIST",
    "pcam": "PCam",
}
PATHMNIST_NCT_LABELS = {
    "ADI": "Adipose",
    "BACK": "Background",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Muscle",
    "NORM": "Normal Colon Mucosa",
    "STR": "Stroma",
    "TUM": "Tumor",
}
MODEL_LABELS = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "uni": "UNI2-h",
    "virchow": "Virchow",
}
EXPLAINER_LABELS = {
    "ig": "IG",
    "gradcam": "Grad-CAM",
    "attnrollout": "AttnRollout",
    "attn_rollout": "AttnRollout",
    "attn_grad": "AttnRollout (Grad)",
    "attnrollout_plain": "AttnRollout",
    "attn_rollout_plain": "AttnRollout",
    "vitshapley": "ViTShapley",
}
EXCLUDED_EXPLAINERS = set()
IMG_INTERP = "bicubic"


# Set random seeds for reproducible runs.
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Infer the class count from the dataset name.
def infer_num_classes(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name == "pcam":
        return 2
    if name == "bloodmnist":
        return 8
    if name == "pathmnist":
        return 9
    raise ValueError(f"Unknown dataset for num_classes inference: {dataset_name}")


# Load a saved model checkpoint and build the model.
def load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    model_name: str,
    dataset_name: str,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    num_classes = ckpt.get("num_classes", None)
    ckpt_model_name = args.get("model", None)
    if ckpt_model_name and ckpt_model_name != model_name:
        print(
            f"[warn] ckpt model '{ckpt_model_name}' != --models '{model_name}' "
            f"for {ckpt_path}; using --models value."
        )

    if num_classes is None:
        num_classes = infer_num_classes(dataset_name)
    if model_name == "resnet18":
        from src.models.cnn import make_resnet18
        model = make_resnet18(num_classes=num_classes, pretrained=False)
    elif model_name == "efficientnet_b0":
        from src.models.cnn import make_efficientnet_b0
        model = make_efficientnet_b0(num_classes=num_classes, pretrained=False)
    elif model_name == "uni":
        from src.models.cnn import make_uni
        model = make_uni(num_classes=num_classes, pretrained=False, freeze_backbone=True)
    elif model_name == "virchow":
        from src.models.cnn import make_virchow
        model = make_virchow(num_classes=num_classes, pretrained=False, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model in ckpt args: {model_name}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, model_name, int(num_classes)


def _label_to_int(y):
    return int(np.array(y).reshape(-1)[0])


def tensor_to_rgb_np(x: torch.Tensor):
    x = x.detach().cpu().squeeze(0)
    x = torch.clamp(x, 0, 1)
    x = x.permute(1, 2, 0).numpy()
    return x


def compute_baseline_tensor(x: torch.Tensor, baseline: str) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if baseline == "zero":
        return torch.zeros_like(x)
    if baseline == "mean":
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x).detach()
    if baseline == "blur":
        k = 11
        pad = k // 2
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad).detach()
    if baseline == "blur21":
        k = 21
        pad = k // 2
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad).detach()
    raise ValueError(f"Unknown baseline: {baseline}")


# Save a small grid showing chosen baselines.
def save_baseline_debug(samples, outdir: Path, baseline: str, dataset_name: str, max_n: int):
    if max_n <= 0:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for i, (idx, x, _) in enumerate(samples[:max_n]):
        x1 = x.unsqueeze(0) if x.ndim == 3 else x
        b = compute_baseline_tensor(x1, baseline)
        rgb = tensor_to_rgb_np(x1)
        b_rgb = tensor_to_rgb_np(b)
        plt.imsave(outdir / f"{dataset_name}_idx{idx}_input.png", rgb)
        plt.imsave(outdir / f"{dataset_name}_idx{idx}_baseline_{baseline}.png", b_rgb)


def sal_to_np(sal_1x1: torch.Tensor):
    s = sal_1x1.detach().cpu().squeeze().numpy()
    s = np.clip(s, 0, 1)
    return s


@torch.no_grad()
def get_pred_target(model, x):
    logits = model(x)
    pred = int(torch.argmax(logits, dim=1).item())
    return pred, logits


def _is_attn_explainer(explainer: str) -> bool:
    return explainer in ["attnrollout", "attn_rollout", "attn_grad",
                         "attnrollout_plain", "attn_rollout_plain"]


def _supports_attn_rollout(model_name: str) -> bool:
    return model_name in ["uni", "virchow"]

def _supports_vitshapley(model_name: str) -> bool:
    return model_name in ["uni", "virchow"]


def _supported_explainers_for_model(model_name: str, explainer_list):
    keep = []
    for explainer in explainer_list:
        if _is_attn_explainer(explainer) and not _supports_attn_rollout(model_name):
            continue
        if explainer == "vitshapley" and not _supports_vitshapley(model_name):
            continue
        keep.append(explainer)
    return keep


# Compute one saliency map for plotting.
def compute_saliency(
    model, x, target, explainer,
    baseline="mean", ig_steps=32, sg_samples=12, sg_sigma=0.10,
    vs_grid=14, vs_perms=32, vs_step=1, vs_baseline="mean", vs_score_mode="logit", vs_normalize=True,
    score_type="logit",
    vs_explainer=None, vs_prefix=None, vs_cfg=None,
):
    if explainer == "ig":
        attr = ig_attribution(model, x, target, baseline=baseline, n_steps=ig_steps)
        return reduce_attribution(attr)
        return reduce_attribution(attr)
    elif explainer == "gradcam":
        cam = gradcam_attribution(model, x, target)
        sal = normalize_map(reduce_attribution(cam))
        return sal
    elif explainer in ["attnrollout", "attn_rollout", "attn_grad"]:
        cam = attention_rollout_attribution(model, x, target, use_grad=True)
        sal = normalize_map(reduce_attribution(cam))
        return sal
    elif explainer in ["attnrollout_plain", "attn_rollout_plain"]:
        cam = attention_rollout_attribution(model, x, target, use_grad=False)
        sal = normalize_map(reduce_attribution(cam))
        return sal
    elif explainer == "vitshapley":
        if vs_explainer is None or vs_prefix is None:
            raise ValueError("vitshapley requires vs_explainer and vs_prefix.")
        if vs_cfg is None:
            vs_cfg = ViTShapleyConfig()
        attr = vitshapley_attribution(model, x, vs_explainer, vs_prefix, target=target, cfg=vs_cfg)
        sal = normalize_map(reduce_attribution(attr))
        return sal
    else:
        raise ValueError(f"Unknown explainer: {explainer}")


# Load the ViTShapley explainer for one model.
def load_vitshapley_for_model(model, model_name, num_classes, ckpt_path, device):
    if not _supports_vitshapley(model_name):
        raise ValueError(f"vitshapley not supported for model '{model_name}'")
    embed_dim = getattr(model.backbone, "num_features", None)
    if embed_dim is None:
        raise ValueError("Backbone missing num_features; cannot load ViTShapley explainer.")
    explainer = ViTShapleyExplainer(embed_dim=embed_dim, num_classes=int(num_classes))
    explainer.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    explainer.to(device).eval()
    prefix = num_prefix_tokens(model.backbone)
    cfg = ViTShapleyConfig()
    return explainer, prefix, cfg


def _get_label_name(args, ds, y_int: int) -> str:
    ds_name = args.dataset.lower()

    if ds_name in ["pathmnist", "bloodmnist"]:
        base_ds = getattr(ds, "dataset", ds)
        classes = getattr(base_ds, "classes", None)
        if ds_name == "pathmnist" and classes:
            if 0 <= int(y_int) < len(classes):
                key = str(classes[int(y_int)])
                return PATHMNIST_NCT_LABELS.get(key, key)

        label_map = getattr(ds, "info", {}).get("label", None)
        if isinstance(label_map, dict):
            key = str(int(y_int))
            if key in label_map:
                return str(label_map[key])
        return f"class_{y_int}"

    if ds_name in ["pcam", "patchcamelyon"]:
        return "tumor" if int(y_int) == 1 else "non-tumor"

    return str(y_int)


# Save a grid of samples, models, and heatmaps.
def save_multi_model_grid(out_path, rgbs, heats_by_model, explainer_list, row_labels, model_list,
                          heat_alpha, heat_alpha_ig, heat_gamma, heat_abs, clip_lo, clip_hi,
                          heat_floor, blur_sigma, bbox_tight=False, save_pdf=False):
    n = len(rgbs)
    explainers_by_model = {
        model: _supported_explainers_for_model(model, explainer_list)
        for model in model_list
    }
    ncols = 1 + sum(len(explainers_by_model[model]) for model in model_list)
    tile = 3.0
    fig = plt.figure(figsize=(tile * ncols, tile * n))

    for i in range(n):
        ax = plt.subplot(n, ncols, i * ncols + 1)
        ax.imshow(rgbs[i], interpolation=IMG_INTERP)
        ax.set_title("input" if i == 0 else "")
        ax.axis("off")
        ax.text(
            0.02, 0.98,
            row_labels[i],
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=20,
            color="black",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0),
            clip_on=True,
        )

        col = 2
        for model in model_list:
            for ex in explainers_by_model[model]:
                ax = plt.subplot(n, ncols, i * ncols + col)
                ax.imshow(rgbs[i], interpolation=IMG_INTERP)
                heat = heats_by_model[model][ex][i]
                heat = _apply_heatmap_viz(heat, heat_abs, clip_lo, clip_hi, heat_gamma, heat_floor)
                if ex in {"ig"} and blur_sigma > 0:
                    heat = _blur_heatmap(heat, blur_sigma)
                alpha = _alpha_for_explainer(ex, heat, heat_alpha, heat_alpha_ig)
                ax.imshow(heat, cmap="jet", vmin=0, vmax=1, alpha=alpha, interpolation=IMG_INTERP)
                if i == 0:
                    model_lbl = MODEL_LABELS.get(model, model)
                    ex_lbl = EXPLAINER_LABELS.get(ex, ex)
                    ax.set_title(f"{model_lbl}\n{ex_lbl}", fontsize=20)
                ax.axis("off")
                col += 1

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04, wspace=0.001, hspace=0.08)
    _save_figure(fig, out_path, bbox_tight=bbox_tight, save_pdf=save_pdf)
    plt.close(fig)


def _apply_heatmap_viz(heat, heat_abs, clip_lo, clip_hi, heat_gamma, heat_floor):
    h = heat.astype(np.float32)
    if heat_abs:
        h = np.abs(h)
    if np.isfinite(h).any():
        lo = np.nanpercentile(h, clip_lo)
        hi = np.nanpercentile(h, clip_hi)
        if np.isclose(hi, lo):
            h = np.clip(h, 0, 1)
        else:
            h = np.clip(h, lo, hi)
            h = (h - lo) / (hi - lo)
    h = np.clip(h, 0, 1)
    if heat_gamma != 1.0:
        h = h ** heat_gamma
    if heat_floor and heat_floor > 0.0:
        h = np.where(h < heat_floor, 0.0, h)
    return h


def _gaussian_kernel1d(sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    radius = max(1, int(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _blur_heatmap(heat: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return heat
    h = torch.from_numpy(heat.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    k1 = _gaussian_kernel1d(sigma)
    kx = k1.view(1, 1, 1, -1)
    ky = k1.view(1, 1, -1, 1)
    pad = (kx.shape[-1] // 2, kx.shape[-1] // 2, ky.shape[-2] // 2, ky.shape[-2] // 2)
    h = F.pad(h, pad, mode="reflect")
    h = F.conv2d(h, kx)
    h = F.conv2d(h, ky)
    return h.squeeze(0).squeeze(0).numpy()


def _alpha_for_explainer(explainer: str, heat: np.ndarray, heat_alpha: float, heat_alpha_ig: float) -> np.ndarray:
    alpha = heat_alpha * heat_alpha_ig if explainer in {"ig"} else heat_alpha
    return np.clip(heat * alpha, 0, 1)


def _save_figure(fig, out_path, bbox_tight=False, save_pdf=False):
    fig.savefig(out_path, dpi=220, bbox_inches="tight" if bbox_tight else None)
    if save_pdf:
        fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight" if bbox_tight else None)


def save_per_image_grid(out_path, rgb, heats_by_model, explainer_list, model_list, title_label,
                        heat_alpha, heat_alpha_ig, heat_gamma, heat_abs, clip_lo, clip_hi, heat_floor,
                        blur_sigma, bbox_tight=False, save_pdf=False):
    nrows = len(model_list)
    ncols = 1 + len(explainer_list)
    tile = 3.0
    fig = plt.figure(figsize=(tile * ncols, tile * nrows))

    for r, model in enumerate(model_list):
        ax = plt.subplot(nrows, ncols, r * ncols + 1)
        ax.imshow(rgb, interpolation=IMG_INTERP)
        if r == 0:
            ax.set_title("input")
        ax.axis("off")

        for c, ex in enumerate(explainer_list):
            ax = plt.subplot(nrows, ncols, r * ncols + 2 + c)
            if _is_attn_explainer(ex) and not _supports_attn_rollout(model):
                ax.set_facecolor("black")
            else:
                ax.imshow(rgb, interpolation=IMG_INTERP)
                heat = heats_by_model[model][ex]
                heat = _apply_heatmap_viz(heat, heat_abs, clip_lo, clip_hi, heat_gamma, heat_floor)
                if ex in {"ig"} and blur_sigma > 0:
                    heat = _blur_heatmap(heat, blur_sigma)
                alpha = _alpha_for_explainer(ex, heat, heat_alpha, heat_alpha_ig)
                ax.imshow(heat, cmap="jet", vmin=0, vmax=1, alpha=alpha, interpolation=IMG_INTERP)
            if r == 0:
                ax.set_title(EXPLAINER_LABELS.get(ex, ex), fontsize=20)
            ax.axis("off")

    for r, model in enumerate(model_list):
        y = 1.0 - (r + 0.5) / nrows
        fig.text(0.01, y, MODEL_LABELS.get(model, model), va="center", ha="left", fontsize=20)

    if title_label:
        fig.suptitle(title_label, y=1.02, fontsize=20)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.04, wspace=0.001, hspace=0.08)
    _save_figure(fig, out_path, bbox_tight=bbox_tight, save_pdf=save_pdf)
    plt.close(fig)


# Save the combined multi-dataset heatmap figure.
def save_all_datasets_grid(out_path, blocks, explainer_list, model_list,
                           heat_alpha, heat_alpha_ig, heat_gamma, heat_abs, clip_lo, clip_hi, heat_floor,
                           blur_sigma, bbox_tight=False, save_pdf=False):
    ncols = 1 + len(explainer_list)
    rows_per_sample = len(model_list)
    nrows = sum(len(b["rgbs"]) * rows_per_sample for b in blocks)
    tile = 3.0
    fig = plt.figure(figsize=(tile * ncols, max(7.0, tile * nrows)))

    row_start = 0
    for block in blocks:
        ds = block["dataset"]
        rgbs = block["rgbs"]
        labels = block["labels"]
        heats = block["heats_by_model"]
        block_rows = len(rgbs) * rows_per_sample

        y_mid = 1.0 - (row_start + block_rows / 2.0) / nrows
        fig.text(0.005, y_mid, DATASET_LABELS.get(ds, ds), rotation=90,
                 va="center", ha="left", fontsize=20)

        for s_idx, rgb in enumerate(rgbs):
            for m_idx, model in enumerate(model_list):
                row = row_start + s_idx * rows_per_sample + m_idx
                y_row = 1.0 - (row + 0.5) / nrows
                fig.text(0.03, y_row, MODEL_LABELS.get(model, model),
                         va="center", ha="left", fontsize=20)

                ax = plt.subplot(nrows, ncols, row * ncols + 1)
                ax.imshow(rgb, interpolation=IMG_INTERP)
                if m_idx == 0:
                    if row_start == 0 and s_idx == 0:
                        ax.set_title("input")
                    ax.text(
                        0.02, 0.98, labels[s_idx],
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=12, color="black",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                    )
                ax.axis("off")

                for c, ex in enumerate(explainer_list):
                    ax = plt.subplot(nrows, ncols, row * ncols + 2 + c)
                    if _is_attn_explainer(ex) and not _supports_attn_rollout(model):
                        pass
                    else:
                        ax.imshow(rgb, interpolation=IMG_INTERP)
                        heat = heats[model][ex][s_idx]
                        heat = _apply_heatmap_viz(heat, heat_abs, clip_lo, clip_hi, heat_gamma, heat_floor)
                        if ex in {"ig"} and blur_sigma > 0:
                            heat = _blur_heatmap(heat, blur_sigma)
                        alpha = _alpha_for_explainer(ex, heat, heat_alpha, heat_alpha_ig)
                        ax.imshow(heat, cmap="jet", vmin=0, vmax=1, alpha=alpha, interpolation=IMG_INTERP)
                    if row_start == 0 and s_idx == 0 and m_idx == 0:
                        ax.set_title(EXPLAINER_LABELS.get(ex, ex), fontsize=20)
                    ax.axis("off")

        row_start += block_rows

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.02, wspace=0.001, hspace=0.02)
    _save_figure(fig, out_path, bbox_tight=bbox_tight, save_pdf=save_pdf)
    plt.close(fig)


# Save the heatmap layout with models as rows.
def save_models_as_rows_multi_samples(out_path, rgbs, heats_by_model, explainer_list, model_list,
                                      row_labels, heat_alpha, heat_alpha_ig, heat_gamma, heat_abs,
                                      clip_lo, clip_hi, heat_floor, blur_sigma, bbox_tight=False,
                                      save_pdf=False):
    n_samples = len(rgbs)
    nrows = n_samples * len(model_list)
    ncols = 1 + len(explainer_list)
    tile = 3.0
    fig = plt.figure(figsize=(tile * ncols, max(7.0, tile * nrows)))

    row = 0
    for i in range(n_samples):
        for m_idx, model in enumerate(model_list):
            ax = plt.subplot(nrows, ncols, row * ncols + 1)
            ax.imshow(rgbs[i], interpolation=IMG_INTERP)
            if m_idx == 0:
                ax.text(
                    0.02, 0.98, row_labels[i],
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=12, color="black",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                )
                if row == 0:
                    ax.set_title("input")
            ax.axis("off")

            for c, ex in enumerate(explainer_list):
                ax = plt.subplot(nrows, ncols, row * ncols + 2 + c)
                if _is_attn_explainer(ex) and not _supports_attn_rollout(model):
                    ax.set_facecolor("black")
                else:
                    ax.imshow(rgbs[i], interpolation=IMG_INTERP)
                    heat = heats_by_model[model][ex][i]
                    heat = _apply_heatmap_viz(heat, heat_abs, clip_lo, clip_hi, heat_gamma, heat_floor)
                    if ex in {"ig"} and blur_sigma > 0:
                        heat = _blur_heatmap(heat, blur_sigma)
                    alpha = _alpha_for_explainer(ex, heat, heat_alpha, heat_alpha_ig)
                    ax.imshow(heat, cmap="jet", vmin=0, vmax=1, alpha=alpha, interpolation=IMG_INTERP)
                if row == 0:
                    ax.set_title(EXPLAINER_LABELS.get(ex, ex), fontsize=20)
                ax.axis("off")

            row += 1

    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.02, wspace=0.001, hspace=0.02)
    _save_figure(fig, out_path, bbox_tight=bbox_tight, save_pdf=save_pdf)
    plt.close(fig)


def get_dataset_by_name(ds_name, args):
    interp = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }.get(args.resize_interp, InterpolationMode.BILINEAR)

    tf_rgb = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=interp),
        transforms.ToTensor(),
    ])

    tf_gray_to_rgb = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=interp),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.ndim == 3 and t.shape[0] == 1 else t),
    ])

    ds_name = ds_name.lower()

    if ds_name == "pathmnist":
        return PathMNIST(split=args.split, transform=tf_rgb, download=args.download)
    if ds_name == "bloodmnist":
        return BloodMNIST(split=args.split, transform=tf_rgb, download=args.download)
    if ds_name in ["pcam", "patchcamelyon"]:
        return PCAM(root=args.pcam_root, split=args.split, transform=tf_rgb, download=args.download)

    raise ValueError(f"Unknown dataset: {ds_name}")


def get_dataset(args):
    return get_dataset_by_name(args.dataset, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts",
                    help="Comma-separated checkpoint paths, one per model")
    ap.add_argument("--models", required=True,
                    help="Comma-separated model names, aligned with --ckpts")
    ap.add_argument("--dataset", default="pathmnist",
                    choices=["pathmnist", "bloodmnist", "pcam"])
    ap.add_argument("--datasets", default="bloodmnist,pathmnist,pcam",
                    help="Comma-separated datasets for layout=all_datasets")
    ap.add_argument("--download", action="store_true", help="download if supported (MedMNIST datasets)")
    ap.add_argument("--pcam_root", default="./pcam",
                    help="PCAM root folder (PCAM will create/use <root>/pcam)")
    ap.add_argument("--outdir", default="./outputs/heatmaps/summary")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--n_per_dataset", type=int, default=3,
                    help="Number of images per dataset for layout=all_datasets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_seed", type=int, default=None,
                    help="Seed used in checkpoint run names (defaults to --seed)")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--resize_interp", default="bicubic",
                    choices=["nearest", "bilinear", "bicubic"],
                    help="Interpolation for resizing small datasets (nearest avoids blur).")
    ap.add_argument("--target_mode", default="pred", choices=["pred", "true"])

    ap.add_argument("--baseline", default="mean", choices=["mean", "zero", "blur", "blur21"])
    ap.add_argument("--ig_steps", type=int, default=64)
    ap.add_argument("--sg_samples", type=int, default=32)
    ap.add_argument("--sg_sigma", type=float, default=0.10)
    ap.add_argument("--score_type", default="logit", choices=["logit", "margin"])

    # ViTShapley
    ap.add_argument("--vs_grid", type=int, default=14)
    ap.add_argument("--vs_perms", type=int, default=32)
    ap.add_argument("--vs_step", type=int, default=1)
    ap.add_argument("--vs_baseline", default="mean", choices=["mean", "zero", "blur"])
    ap.add_argument("--vs_score_mode", default="logit", choices=["logit", "prob"])
    ap.add_argument("--vs_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vs_explainer",
                    help="Path to ViTShapley explainer ckpt (used for all ViT models if --vs_explainers not set)")
    ap.add_argument("--vs_explainers",
                    help="Comma-separated ViTShapley explainer ckpt paths aligned with --models")

    ap.add_argument("--explainers", default="ig,gradcam,attnrollout",
                    help="Comma-separated list of explainers")
    ap.add_argument("--layout", default="samples_as_rows",
                    choices=["samples_as_rows", "models_as_rows", "all_datasets"],
                    help="samples_as_rows = current grid; models_as_rows = one figure per sample")
    ap.add_argument("--models_as_rows_single", action="store_true",
                    help="When layout=models_as_rows, save a single figure with all samples.")
    ap.add_argument("--all_datasets_separate", action="store_true",
                    help="When layout=all_datasets, also save one figure per dataset.")
    ap.add_argument("--all_datasets_only", action="store_true",
                    help="When layout=all_datasets, save only per-dataset figures (no combined grid).")
    ap.add_argument("--runs_root", default="./outputs/runs",
                    help="Runs root for layout=all_datasets")
    ap.add_argument("--heat_alpha", type=float, default=0.95,
                    help="Heatmap overlay alpha (higher makes faint maps more visible).")
    ap.add_argument("--heat_alpha_ig", type=float, default=1.3,
                    help="Extra overlay alpha for IG/SmoothGrad (multiplies heat_alpha).")
    ap.add_argument("--heat_gamma", type=float, default=0.7,
                    help="Gamma correction for heatmaps (<1 boosts low activations).")
    ap.add_argument("--heat_floor", type=float, default=0.15,
                    help="Floor small heat values to 0 to reduce background (0 disables).")
    ap.add_argument("--saliency_blur", type=float, default=0.5,
                    help="Gaussian blur sigma for IG/SmoothGrad heatmaps (0 disables).")
    ap.add_argument("--heat_abs", action="store_true",
                    help="Use absolute value of heatmap before visualization.")
    ap.add_argument("--heat_clip_pct", default="1,99",
                    help="Percentile clip range for heatmaps, e.g., '1,99' or '0,100'.")
    ap.add_argument("--bbox_tight", action=argparse.BooleanOptionalAction, default=False,
                    help="Use bbox_inches='tight' when saving figures (can add whitespace).")
    ap.add_argument("--save_pdf", action="store_true",
                    help="Also save a PDF next to each PNG.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device for attribution computation.")
    ap.add_argument("--save_baseline_n", type=int, default=0,
                    help="Save baseline/input images for first N samples (0 disables).")
    ap.add_argument("--baseline_outdir", default="output_new_debug_ig",
                    help="Output folder for baseline debug images.")

    args = ap.parse_args()
    set_seed(args.seed)
    if args.ckpt_seed is None:
        args.ckpt_seed = args.seed

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ckpts = [p.strip() for p in args.ckpts.split(",") if p.strip()] if args.ckpts else []
    if args.layout != "all_datasets":
        if not ckpts:
            raise ValueError("--ckpts is required unless --layout=all_datasets")
        if len(ckpts) != len(models):
            raise ValueError("--ckpts and --models must have the same length")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explainer_list = [s.strip() for s in args.explainers.split(",") if s.strip()]
    explainer_list = [e for e in explainer_list if e not in EXCLUDED_EXPLAINERS]

    vs_ckpt_map = {}
    if "vitshapley" in explainer_list:
        if args.vs_explainers:
            vs_list = [s.strip() for s in args.vs_explainers.split(",") if s.strip()]
            if len(vs_list) != len(models):
                raise ValueError("--vs_explainers must align with --models length")
            vs_ckpt_map = {m: p for m, p in zip(models, vs_list)}
        elif args.vs_explainer:
            vs_ckpt_map = {m: args.vs_explainer for m in models}
        else:
            raise ValueError("vitshapley requested but no --vs_explainer/--vs_explainers provided")

    clip_lo, clip_hi = 1.0, 99.0
    if args.heat_clip_pct:
        try:
            parts = [float(p.strip()) for p in args.heat_clip_pct.split(",")]
            if len(parts) == 2:
                clip_lo, clip_hi = parts
        except ValueError:
            pass
    def build_samples(ds_name, n_samples, seed):
        ds = get_dataset_by_name(ds_name, args)
        rng = np.random.default_rng(seed)
        if n_samples > len(ds):
            raise ValueError(f"--n {n_samples} is larger than dataset size {len(ds)} for split={args.split}")
        idxs = rng.choice(len(ds), size=n_samples, replace=False)
        samples = []
        rgbs = []
        row_labels = []
        for idx in idxs:
            x, y_true = ds[int(idx)]
            y_true = _label_to_int(y_true)
            if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            samples.append((int(idx), x, int(y_true)))
        for idx, x, y_true in samples:
            rgb = tensor_to_rgb_np(x.unsqueeze(0))
            rgbs.append(rgb)
            true_name = _get_label_name(args, ds, y_true)
            row_labels.append(f"idx {idx}\ntrue: {true_name}")
        return ds, samples, rgbs, row_labels

    def build_ckpts_for_dataset(ds_name):
        ckpt_list = []
        runs_root = Path(args.runs_root)
        for m in models:
            run = f"{ds_name}_{m}_seed{args.ckpt_seed}"
            ckpt = runs_root / run / "ckpt_best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
            ckpt_list.append(str(ckpt))
        return ckpt_list

    def compute_block(ds_name, n_samples, seed):
        ds, samples, rgbs, row_labels = build_samples(ds_name, n_samples, seed)
        if args.save_baseline_n > 0:
            save_baseline_debug(
                samples,
                Path(args.baseline_outdir) / ds_name,
                args.baseline,
                ds_name,
                args.save_baseline_n,
            )
        ckpt_list = build_ckpts_for_dataset(ds_name)
        heats_by_model = {m: {ex: [] for ex in explainer_list} for m in models}
        skipped_attn = set()
        skipped_vit = set()
        for ckpt_path, model_name in zip(ckpt_list, models):
            model, _, num_classes = load_model_from_ckpt(ckpt_path, device, model_name, ds_name)
            vs_explainer = None
            vs_prefix = None
            vs_cfg = None
            if "vitshapley" in explainer_list:
                if _supports_vitshapley(model_name):
                    vs_ckpt = vs_ckpt_map.get(model_name)
                    if not vs_ckpt:
                        raise ValueError(f"Missing vitshapley ckpt for model '{model_name}'")
                    vs_explainer, vs_prefix, vs_cfg = load_vitshapley_for_model(
                        model, model_name, num_classes, vs_ckpt, device
                    )
                else:
                    skipped_vit.add(model_name)
            for idx, x, y_true in samples:
                x = x.unsqueeze(0).to(device)
                pred, logits = get_pred_target(model, x)
                target = pred if args.target_mode == "pred" else y_true
                _ = logits
                for ex in explainer_list:
                    if _is_attn_explainer(ex) and not _supports_attn_rollout(model_name):
                        if (model_name, ex) not in skipped_attn:
                            print(f"[warn] skipping {ex} for model '{model_name}' (no attention)")
                            skipped_attn.add((model_name, ex))
                        h, w = x.shape[-2], x.shape[-1]
                        heat = np.zeros((h, w), dtype=np.float32)
                        heats_by_model[model_name][ex].append(heat)
                        continue
                    if ex == "vitshapley" and not _supports_vitshapley(model_name):
                        if (model_name, ex) not in skipped_vit:
                            print(f"[warn] skipping vitshapley for model '{model_name}' (not a ViT)")
                            skipped_vit.add((model_name, ex))
                        h, w = x.shape[-2], x.shape[-1]
                        heat = np.zeros((h, w), dtype=np.float32)
                        heats_by_model[model_name][ex].append(heat)
                        continue
                    sal = compute_saliency(
                        model, x, target, ex,
                        baseline=args.baseline, ig_steps=args.ig_steps,
                        sg_samples=args.sg_samples, sg_sigma=args.sg_sigma,
                        vs_grid=args.vs_grid, vs_perms=args.vs_perms, vs_step=args.vs_step,
                        vs_baseline=args.vs_baseline, vs_score_mode=args.vs_score_mode,
                        vs_normalize=args.vs_normalize,
                        score_type=args.score_type,
                        vs_explainer=vs_explainer, vs_prefix=vs_prefix, vs_cfg=vs_cfg,
                    )
                    heat = sal_to_np(sal)
                    heats_by_model[model_name][ex].append(heat)
        return {
            "dataset": ds_name,
            "samples": samples,
            "rgbs": rgbs,
            "labels": row_labels,
            "heats_by_model": heats_by_model,
        }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.layout == "all_datasets":
        ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.all_datasets_only:
            for i, ds_name in enumerate(ds_list):
                block = compute_block(ds_name, args.n_per_dataset, args.seed + i)
                out_path = outdir / f"summary_{ds_name}_{args.split}_n{args.n_per_dataset}.png"
                save_all_datasets_grid(out_path, [block], explainer_list, models,
                                       args.heat_alpha, args.heat_alpha_ig, args.heat_gamma, args.heat_abs,
                                       clip_lo, clip_hi,
                                       args.heat_floor, args.saliency_blur, args.bbox_tight, args.save_pdf)
                print("Saved summary heatmap to:", out_path.resolve())
        else:
            blocks = []
            for i, ds_name in enumerate(ds_list):
                blocks.append(compute_block(ds_name, args.n_per_dataset, args.seed + i))
            out_path = outdir / f"summary_all_datasets_{args.split}_n{args.n_per_dataset}.png"
            save_all_datasets_grid(out_path, blocks, explainer_list, models,
                                   args.heat_alpha, args.heat_alpha_ig, args.heat_gamma, args.heat_abs,
                                   clip_lo, clip_hi,
                                   args.heat_floor, args.saliency_blur, args.bbox_tight, args.save_pdf)
            print("Saved summary heatmap to:", out_path.resolve())
            if args.all_datasets_separate:
                for block in blocks:
                    ds_name = block["dataset"]
                    out_path = outdir / f"summary_{ds_name}_{args.split}_n{args.n_per_dataset}.png"
                    save_all_datasets_grid(out_path, [block], explainer_list, models,
                                           args.heat_alpha, args.heat_alpha_ig, args.heat_gamma, args.heat_abs,
                                           clip_lo, clip_hi,
                                           args.heat_floor, args.saliency_blur, args.bbox_tight, args.save_pdf)
                    print("Saved summary heatmap to:", out_path.resolve())
    else:
        ds = get_dataset(args)
        rng = np.random.default_rng(args.seed)
        if args.n > len(ds):
            raise ValueError(f"--n {args.n} is larger than dataset size {len(ds)} for split={args.split}")
        idxs = rng.choice(len(ds), size=args.n, replace=False)
        heats_by_model = {m: {ex: [] for ex in explainer_list} for m in models}
        rgbs = []
        row_labels = []
        skipped_attn = set()

        samples = []
        for idx in idxs:
            x, y_true = ds[int(idx)]
            y_true = _label_to_int(y_true)
            if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            samples.append((int(idx), x, int(y_true)))

        for idx, x, y_true in samples:
            rgb = tensor_to_rgb_np(x.unsqueeze(0))
            rgbs.append(rgb)
            true_name = _get_label_name(args, ds, y_true)
            row_labels.append(f"idx {idx}\ntrue: {true_name}")
        if args.save_baseline_n > 0:
            save_baseline_debug(
                samples,
                Path(args.baseline_outdir) / args.dataset,
                args.baseline,
                args.dataset,
                args.save_baseline_n,
            )

        for ckpt_path, model_name in zip(ckpts, models):
            model, _, num_classes = load_model_from_ckpt(ckpt_path, device, model_name, args.dataset)
            vs_explainer = None
            vs_prefix = None
            vs_cfg = None
            if "vitshapley" in explainer_list:
                if _supports_vitshapley(model_name):
                    vs_ckpt = vs_ckpt_map.get(model_name)
                    if not vs_ckpt:
                        raise ValueError(f"Missing vitshapley ckpt for model '{model_name}'")
                    vs_explainer, vs_prefix, vs_cfg = load_vitshapley_for_model(
                        model, model_name, num_classes, vs_ckpt, device
                    )
                else:
                    skipped_attn.add((model_name, "vitshapley"))
            for idx, x, y_true in samples:
                x = x.unsqueeze(0).to(device)
                pred, logits = get_pred_target(model, x)
                target = pred if args.target_mode == "pred" else y_true
                _ = logits  # keep parity with other scripts if you want to extend labels later

                for ex in explainer_list:
                    if _is_attn_explainer(ex) and not _supports_attn_rollout(model_name):
                        if (model_name, ex) not in skipped_attn:
                            print(f"[warn] skipping {ex} for model '{model_name}' (no attention)")
                            skipped_attn.add((model_name, ex))
                        h, w = x.shape[-2], x.shape[-1]
                        heat = np.zeros((h, w), dtype=np.float32)
                        heats_by_model[model_name][ex].append(heat)
                        continue
                    if ex == "vitshapley" and not _supports_vitshapley(model_name):
                        if (model_name, ex) not in skipped_attn:
                            print(f"[warn] skipping vitshapley for model '{model_name}' (not a ViT)")
                            skipped_attn.add((model_name, ex))
                        h, w = x.shape[-2], x.shape[-1]
                        heat = np.zeros((h, w), dtype=np.float32)
                        heats_by_model[model_name][ex].append(heat)
                        continue
                    sal = compute_saliency(
                        model, x, target, ex,
                        baseline=args.baseline, ig_steps=args.ig_steps,
                        sg_samples=args.sg_samples, sg_sigma=args.sg_sigma,
                        vs_grid=args.vs_grid, vs_perms=args.vs_perms, vs_step=args.vs_step,
                        vs_baseline=args.vs_baseline, vs_score_mode=args.vs_score_mode,
                        vs_normalize=args.vs_normalize,
                        score_type=args.score_type,
                        vs_explainer=vs_explainer, vs_prefix=vs_prefix, vs_cfg=vs_cfg,
                    )
                    heat = sal_to_np(sal)
                    heats_by_model[model_name][ex].append(heat)

        if args.layout == "models_as_rows":
            if args.models_as_rows_single:
                out_path = outdir / f"summary_{args.dataset}_{args.split}_n{args.n}.png"
                save_models_as_rows_multi_samples(
                    out_path, rgbs, heats_by_model, explainer_list, models, row_labels,
                    args.heat_alpha, args.heat_alpha_ig, args.heat_gamma, args.heat_abs, clip_lo, clip_hi,
                    args.heat_floor, args.saliency_blur, args.bbox_tight, args.save_pdf
                )
                print("Saved summary heatmap to:", out_path.resolve())
            else:
                for i, (idx, _, _) in enumerate(samples):
                    per_model = {
                        m: {ex: heats_by_model[m][ex][i] for ex in explainer_list}
                        for m in models
                    }
                    out_path = outdir / f"summary_{args.dataset}_{args.split}_idx{idx}.png"
                    save_per_image_grid(out_path, rgbs[i], per_model, explainer_list, models,
                                        row_labels[i], args.heat_alpha, args.heat_alpha_ig, args.heat_gamma,
                                        args.heat_abs, clip_lo, clip_hi, args.heat_floor,
                                        args.saliency_blur, args.bbox_tight, args.save_pdf)
                    print("Saved summary heatmap to:", out_path.resolve())
        else:
            out_path = outdir / f"summary_{args.dataset}_{args.split}_n{args.n}.png"
            save_multi_model_grid(out_path, rgbs, heats_by_model, explainer_list, row_labels, models,
                                  args.heat_alpha, args.heat_alpha_ig, args.heat_gamma, args.heat_abs,
                                  clip_lo, clip_hi, args.heat_floor, args.saliency_blur, args.bbox_tight,
                                  args.save_pdf)
            print("Saved summary heatmap to:", out_path.resolve())


if __name__ == "__main__":
    main()
