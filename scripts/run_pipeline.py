#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_command(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def run_summary(root: Path, outputs_dir: Path, python_bin: str) -> None:
    summary_dir = outputs_dir / "summary"
    results_root = outputs_dir / "results"
    runs_root = outputs_dir / "runs"
    figs_dir = summary_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            python_bin,
            "scripts/make_summary_all.py",
            "--results_root",
            str(results_root),
            "--runs_root",
            str(runs_root),
            "--explainers",
            "ig,gradcam,attnrollout,vitshapley",
            "--out_csv",
            str(summary_dir / "summary_all.csv"),
            "--per_dataset_dir",
            str(summary_dir),
        ],
        [
            python_bin,
            "src/plotting/plot_explainer_metrics_panel.py",
            "--csv",
            str(summary_dir / "summary_all.csv"),
            "--condition",
            "trained",
            "--out",
            str(figs_dir / "explainer_metrics_panel.png"),
        ],
    ]
    for cmd in commands:
        run_command(cmd, cwd=root)


# Generate summary heatmaps from saved checkpoints.
def run_heatmaps(root: Path, outputs_dir: Path, python_bin: str, seeds: list[str]) -> None:
    summary_dir = outputs_dir / "summary"
    runs_root = outputs_dir / "runs"
    heatmap_dir = summary_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["bloodmnist", "pathmnist", "pcam"]
    models = ["resnet18", "efficientnet_b0", "uni", "virchow"]
    model_to_tag = {"uni": "uni2h", "virchow": "virchow2"}

    for seed in seeds:
        for dataset in datasets:
            ckpts: list[str] = []
            models_ok: list[str] = []
            vs_explainers: list[str] = []
            have_vit = True
            first_vit_ckpt = ""
            vs_baseline = "mean" if dataset == "bloodmnist" else "blur"

            for model in models:
                ckpt = runs_root / f"{dataset}_{model}_seed{seed}" / "ckpt_best.pt"
                if not ckpt.exists():
                    continue
                ckpts.append(str(ckpt))
                models_ok.append(model)

                if model in model_to_tag:
                    vs_ckpt = runs_root / dataset / model_to_tag[model] / "vitshapley" / f"seed{seed}" / "explainer_best.pt"
                    if vs_ckpt.exists():
                        vs_explainers.append(str(vs_ckpt))
                        if not first_vit_ckpt:
                            first_vit_ckpt = str(vs_ckpt)
                    else:
                        have_vit = False
                        vs_explainers.append("")
                else:
                    vs_explainers.append("")

            if not ckpts:
                print(f"[WARN] No checkpoints found for summary heatmaps: dataset={dataset} seed={seed}")
                continue

            explainers = "ig,gradcam,attnrollout,vitshapley"
            cmd = [
                python_bin,
                "src/plotting/make_heatmap_summary.py",
                "--ckpts",
                ",".join(ckpts),
                "--models",
                ",".join(models_ok),
                "--dataset",
                dataset,
                "--split",
                "test",
                "--n",
                "5",
                "--img_size",
                "224",
                "--target_mode",
                "pred",
                "--vs_baseline",
                vs_baseline,
                "--outdir",
                str(heatmap_dir / f"seed{seed}"),
            ]

            if have_vit and first_vit_ckpt:
                vs_explainers = [item or first_vit_ckpt for item in vs_explainers]
                cmd.extend(["--explainers", explainers, "--vs_explainers", ",".join(vs_explainers)])
            else:
                cmd.extend(["--explainers", "ig,gradcam,attnrollout"])

            run_command(cmd, cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utility runner for summary aggregation and heatmap generation."
    )
    parser.add_argument(
        "--stages",
        default="summary",
        help="Comma-separated stages from: summary,heatmaps",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter to use.")
    parser.add_argument("--outputs-dir-name", default="outputs", help="Repo-relative output directory name.")
    parser.add_argument("--heatmap-seeds", default="0,1,2", help="Seeds to use when stage includes heatmaps.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    outputs_dir = root / args.outputs_dir_name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stages = parse_csv(args.stages)
    if "summary" in stages:
        run_summary(root, outputs_dir, args.python_bin)
    if "heatmaps" in stages:
        run_heatmaps(root, outputs_dir, args.python_bin, parse_csv(args.heatmap_seeds))


if __name__ == "__main__":
    main()
