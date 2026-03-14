#!/usr/bin/env python3
"""
Combined figure: rows are metrics (faithfulness, robustness, sanity, fidelity),
columns are models, lines are datasets, x-axis is explainer.

Usage:
  python src/plotting/plot_explainer_metrics_panel.py \
    --csv outputs/summary/summary_all1.csv \
    --condition trained \
    --out outputs/summary/figs/explainer_metrics_panel.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PREF_EXPLAINERS = [
    "ig", "gradcam",
    "attnrollout", "vitshapley",
]
EXCLUDED_EXPLAINERS = set()
DATASET_LABELS = {
    "bloodmnist": "BloodMNIST",
    "pathmnist": "PathMNIST",
    "pcam": "PCam",
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
    "vitshapley": "ViTShapley",
}


def ordered_explainers(explainers):
    ex = [e for e in PREF_EXPLAINERS if e in explainers] + [e for e in explainers if e not in PREF_EXPLAINERS]
    return ex

def labelize(values, mapping):
    return [mapping.get(v, v) for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to summary_all*.csv")
    ap.add_argument("--condition", default="trained", choices=["trained"])
    ap.add_argument("--datasets", default="bloodmnist,pathmnist,pcam")
    ap.add_argument("--models", default="resnet18,efficientnet_b0,uni,virchow")
    ap.add_argument("--out", default="./outputs/summary/figs/explainer_metrics_panel.png")
    ap.add_argument("--save_pdf", action="store_true",
                    help="Also save a PDF next to the output image.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    df["explainer"] = df["explainer"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().str.lower()

    df = df[df["condition"] == args.condition]
    df = df[~df["explainer"].isin(EXCLUDED_EXPLAINERS)]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    df = df[df["dataset"].isin(datasets) & df["model"].isin(models)]

    # Metric selection
    sanity_col = None
    if "sanity_rand_last25_mean" in df.columns:
        sanity_col = "sanity_rand_last25_mean"
        df["sanity_sensitivity"] = 1.0 - pd.to_numeric(df[sanity_col], errors="coerce")
        sanity_label = "Sanity (1 - rand_last25)"
        sanity_metric = "sanity_sensitivity"
    elif "sanity_original_mean" in df.columns:
        sanity_metric = "sanity_original_mean"
        sanity_label = "Sanity (original sim)"
    else:
        sanity_metric = None
        sanity_label = "Sanity (missing)"

    metrics = [
        ("faith_ins_mean", "Faithfulness insertion AUC"),
        ("faith_del_mean", "Faithfulness deletion AUC"),
        ("rob_jitter_samepred_mean", "Robustness (jitter samepred)"),
        (sanity_metric, sanity_label),
        ("fid_corr_mean", "Fidelity (corr)"),
    ]

    metrics = [(m, label) for m, label in metrics if m is not None and m in df.columns]

    explainers = ordered_explainers(sorted(df["explainer"].dropna().unique().tolist()))
    explainer_labels = labelize(explainers, EXPLAINER_LABELS)

    n_rows = len(metrics)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows), sharey=False)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, (metric, label) in enumerate(metrics):
        for j, model in enumerate(models):
            ax = axes[i][j]
            any_line = False
            for ds in datasets:
                sub = df[(df["dataset"] == ds) & (df["model"] == model)]
                if sub.empty:
                    continue
                agg = sub.groupby("explainer", dropna=False)[metric].mean(numeric_only=True).reset_index()
                agg = agg.set_index("explainer").reindex(explainers).reset_index()
                ax.plot(agg["explainer"], agg[metric], marker="o", label=DATASET_LABELS.get(ds, ds))
                any_line = True

            if not any_line:
                ax.set_title(f"{model} (no data)")
                ax.axis("off")
                continue

            if i == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            if j == 0:
                ax.set_ylabel(label)
            ax.set_xlabel("explainer")
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
            ax.set_xticklabels(explainer_labels, rotation=25, ha="right")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, 1.03))

    fig.suptitle(f"Explainer metrics by model (condition: {args.condition})", y=1.08)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if args.save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path.resolve())
    if args.save_pdf:
        print("Saved:", pdf_path.resolve())


if __name__ == "__main__":
    main()
