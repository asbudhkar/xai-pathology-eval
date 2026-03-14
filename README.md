# XAI Pathology Eval

Code for evaluating explainability methods on computational pathology image classification models.

This repository supports training CNN and ViT-based pathology classifiers, computing explanation quality metrics, training ViTShapley explainers for transformer backbones, and generating the summary figures used in the benchmark.

### Overview

- Datasets: `bloodmnist`, `pathmnist`, `pcam`
- Models: `resnet18`, `efficientnet_b0`, `uni`, `virchow`
- Explainers: `ig`, `gradcam`, `attnrollout`, `vitshapley`

Important note:
- In this repo, `pathmnist` is loaded from NCT-CRC-HE-100K and split into train/val/test with a fixed `80/10/10` split, not the official MedMNIST PathMNIST split.

### Main Files

- `scripts/train.py`: train one classifier checkpoint
- `scripts/run_pipeline.py`: utility runner for summary aggregation and heatmaps
- `scripts/gen_eval_indices.py`: generate fixed stratified evaluation indices
- `scripts/eval_faithfulness.py`
- `scripts/eval_fidelity.py`
- `scripts/eval_robustness.py`
- `scripts/eval_sanity_weights.py`
- `scripts/train_masked_finetune.py`
- `scripts/train_vitshapley_explainer.py`
- `scripts/train_vitshapley_surrogate.py`
- `scripts/make_summary_all.py`
- `src/plotting/plot_explainer_metrics_panel.py`
- `src/plotting/make_heatmap_summary.py`

### Output Layout

All default paths are repo-relative:

- `./outputs/runs`: checkpoints and `metrics.json`
- `./outputs/results`: explanation metric outputs
- `./outputs/summary`: aggregated CSVs and figures
- `./outputs/indices`: fixed evaluation index CSVs

### Folder Structure

Recommended layout:

```text
xai-pathology-eval/
├── scripts/
├── src/
├── outputs/
│   ├── indices/
│   ├── results/
│   ├── runs/
│   └── summary/
├── pcam/
└── data/
    └── pathmnist/
        └── NCT-CRC-HE-100K/
```

What goes where:

- `./outputs/runs/`
  classifier checkpoints and training metrics
  example: `./outputs/runs/bloodmnist_resnet18_seed0/ckpt_best.pt`
- `./outputs/results/`
  explanation metric outputs for each trained run
  example: `./outputs/results/bloodmnist_resnet18_seed0/faithfulness_ig_summary.json`
- `./outputs/summary/`
  aggregated CSVs and final figures
  example: `./outputs/summary/summary_all.csv`
- `./outputs/indices/`
  fixed stratified evaluation subsets
  example: `./outputs/indices/eval_indices_bloodmnist_normal.csv`
- `./pcam/`
  PCam data root used by the scripts
- `./data/pathmnist/NCT-CRC-HE-100K/`
  colorectal histology dataset used by the repo's `pathmnist` loader

Dataset placement:

- `bloodmnist`
  downloaded through `medmnist` into its standard cache location; no manual repo folder is required
- `pcam`
  expected under `./pcam` unless a different `--pcam_root` is provided
- `pathmnist`
  expected at `./data/pathmnist/NCT-CRC-HE-100K`

### Environment

Run commands from the repo root:

```bash
cd /path/to/xai-pathology-eval
```

The project expects a Python environment with PyTorch, torchvision, medmnist, pandas, numpy, and matplotlib installed.

### Standard Workflow

Recommended order:

1. Train classifier checkpoints.
2. Generate fixed 300-image evaluation indices.
3. Run explanation metric evaluation.
4. Aggregate summary tables.
5. Generate the combined metrics panel and heatmaps.
6. Optionally train ViTShapley checkpoints for `uni` and `virchow`.

### 1. Train Classifiers

Single run example:

```bash
python3 scripts/train.py \
  --dataset bloodmnist \
  --model resnet18 \
  --seed 0 \
  --epochs 6 \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 4
```

This writes:

- `./outputs/runs/bloodmnist_resnet18_seed0/ckpt_best.pt`
- `./outputs/runs/bloodmnist_resnet18_seed0/metrics.json`

For multiple runs, repeat `scripts/train.py` across datasets, models, and seeds in your own scheduler or cluster setup.

### 2. Generate Fixed Evaluation Indices

Generate the same stratified 300-image subset for each dataset:

```bash
python3 scripts/gen_eval_indices.py --dataset bloodmnist --n 300 --seed 0 --img_size 224 --outdir ./outputs/indices
python3 scripts/gen_eval_indices.py --dataset pathmnist --n 300 --seed 0 --img_size 224 --outdir ./outputs/indices
python3 scripts/gen_eval_indices.py --dataset pcam --n 300 --seed 0 --img_size 224 --outdir ./outputs/indices
```

This writes files such as:

- `./outputs/indices/eval_indices_bloodmnist_normal.csv`

### 3. Run Explanation Metrics

Each evaluation script reads a classifier checkpoint and writes outputs into `./outputs/results/<run_name>/`.

Example faithfulness run:

```bash
python3 scripts/eval_faithfulness.py \
  --ckpt ./outputs/runs/bloodmnist_resnet18_seed0/ckpt_best.pt \
  --outdir ./outputs/results/bloodmnist_resnet18_seed0 \
  --explainer ig \
  --indices_in ./outputs/indices/eval_indices_bloodmnist_normal.csv \
  --n 300 \
  --target_mode pred \
  --correct_only \
  --baseline mean \
  --fill blur
```

Example fidelity run:

```bash
python3 scripts/eval_fidelity.py \
  --ckpt ./outputs/runs/bloodmnist_resnet18_seed0/ckpt_best.pt \
  --outdir ./outputs/results/bloodmnist_resnet18_seed0 \
  --explainer ig \
  --indices_in ./outputs/indices/eval_indices_bloodmnist_normal.csv \
  --n 300 \
  --target_mode pred \
  --correct_only \
  --baseline mean \
  --fill blur
```

Example robustness run:

```bash
python3 scripts/eval_robustness.py \
  --ckpt ./outputs/runs/bloodmnist_resnet18_seed0/ckpt_best.pt \
  --outdir ./outputs/results/bloodmnist_resnet18_seed0 \
  --explainer ig \
  --indices_in ./outputs/indices/eval_indices_bloodmnist_normal.csv \
  --n 300 \
  --target_mode pred \
  --correct_only \
  --baseline mean
```

Example sanity run:

```bash
python3 scripts/eval_sanity_weights.py \
  --ckpt ./outputs/runs/bloodmnist_resnet18_seed0/ckpt_best.pt \
  --outdir ./outputs/results/bloodmnist_resnet18_seed0 \
  --explainer ig \
  --indices_in ./outputs/indices/eval_indices_bloodmnist_normal.csv \
  --n 50 \
  --target_mode pred \
  --correct_only \
  --baseline mean
```

For multiple runs, repeat the evaluation scripts across checkpoints, explainers, and seeds in your own scheduler or cluster setup.

### 4. Aggregate Results

Manual aggregation:

```bash
python3 scripts/make_summary_all.py \
  --results_root ./outputs/results \
  --runs_root ./outputs/runs \
  --explainers ig,gradcam,attnrollout,vitshapley \
  --out_csv ./outputs/summary/summary_all.csv \
  --per_dataset_dir ./outputs/summary
```

This writes:

- `./outputs/summary/summary_all.csv`

### 5. Plot Summary Figures

```bash
python3 src/plotting/plot_explainer_metrics_panel.py \
  --csv ./outputs/summary/summary_all.csv \
  --condition trained \
  --out ./outputs/summary/figs/explainer_metrics_panel.png
```

### 6. Generate Final Heatmap Figures

The final paper-style heatmap summary figure is produced by `src/plotting/make_heatmap_summary.py`.

Supported figure behavior:

- `resnet18` and `efficientnet_b0`: `ig`, `gradcam`
- `uni` and `virchow`: `ig`, `gradcam`, `attnrollout`, `vitshapley`

Example:

```bash
CUDA_VISIBLE_DEVICES=7 python3 src/plotting/make_heatmap_summary.py \
  --ckpts ./outputs/runs/bloodmnist_resnet18_seed0/ckpt_best.pt,./outputs/runs/bloodmnist_efficientnet_b0_seed0/ckpt_best.pt,./outputs/runs/bloodmnist_uni_seed0/ckpt_best.pt,./outputs/runs/bloodmnist_virchow_seed0/ckpt_best.pt \
  --models resnet18,efficientnet_b0,uni,virchow \
  --dataset bloodmnist \
  --split test \
  --n 5 \
  --img_size 224 \
  --target_mode pred \
  --vs_baseline mean \
  --explainers ig,gradcam,attnrollout,vitshapley \
  --vs_explainers ./outputs/runs/bloodmnist/uni2h/vitshapley/seed0/explainer_best.pt,./outputs/runs/bloodmnist/uni2h/vitshapley/seed0/explainer_best.pt,./outputs/runs/bloodmnist/uni2h/vitshapley/seed0/explainer_best.pt,./outputs/runs/bloodmnist/virchow2/vitshapley/seed0/explainer_best.pt \
  --save_pdf \
  --outdir ./outputs/summary/heatmaps_manual/bloodmnist_seed0
```

### 7. Hugging Face Access for UNI2-h and Virchow

The pretrained `uni` and `virchow` backbones are loaded from Hugging Face through `timm`. To use `pretrained=True`, you may need approved access to the corresponding model repositories and a valid Hugging Face token.

Set one of these environment variables before running training or evaluation with pretrained ViT backbones:

```bash
export HF_TOKEN=your_huggingface_token
```

or

```bash
export HUGGINGFACE_HUB_TOKEN=your_huggingface_token
```

If the weights are already cached locally, the token may not be needed on later runs. If you use `pretrained=False`, this access path is not used.

Or through the generic runner:

```bash
python3 scripts/run_pipeline.py --stages heatmaps --heatmap-seeds 0,1,2
```

### ViTShapley Workflow

ViTShapley is only used for `uni` and `virchow`.

Typical order:

1. Optional surrogate training
2. Optional masked fine-tuning
3. ViTShapley explainer training

Optional surrogate:

```bash
python3 scripts/train_vitshapley_surrogate.py \
  --ckpt ./outputs/runs/bloodmnist_uni_seed0/ckpt_best.pt \
  --outdir ./outputs/runs/bloodmnist/uni2h/vitshapley_surrogate/seed0 \
  --dataset bloodmnist \
  --img_size 224 \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-4 \
  --device cuda
```

Optional masked fine-tuning:

```bash
python3 scripts/train_masked_finetune.py \
  --ckpt ./outputs/runs/bloodmnist_uni_seed0/ckpt_best.pt \
  --outdir ./outputs/runs/bloodmnist/uni2h/masked_finetune/seed0 \
  --dataset bloodmnist \
  --img_size 224 \
  --batch_size 64 \
  --epochs 5 \
  --unfreeze_blocks 2 \
  --device cuda
```

Explainer training:

```bash
python3 scripts/train_vitshapley_explainer.py \
  --ckpt ./outputs/runs/bloodmnist/uni2h/masked_finetune/seed0/ckpt_best.pt \
  --outdir ./outputs/runs/bloodmnist/uni2h/vitshapley/seed0 \
  --dataset bloodmnist \
  --img_size 224 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-4 \
  --score_mode logit \
  --target_mode pred \
  --device cuda
```

With surrogate scoring:

```bash
python3 scripts/train_vitshapley_explainer.py \
  --ckpt ./outputs/runs/bloodmnist/uni2h/masked_finetune/seed0/ckpt_best.pt \
  --outdir ./outputs/runs/bloodmnist/uni2h/vitshapley/seed0 \
  --dataset bloodmnist \
  --img_size 224 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-4 \
  --score_mode logit \
  --target_mode pred \
  --surrogate_ckpt ./outputs/runs/bloodmnist/uni2h/vitshapley_surrogate/seed0/surrogate_best.pt \
  --device cuda
```

Expected output:

- `./outputs/runs/<dataset>/<uni2h|virchow2>/vitshapley/seed<seed>/explainer_best.pt`

### Notes

- Use the same `--indices_in` file across explainers for reproducible comparison.
- ViTShapley uses an internal all-zero reference for explainer training and efficiency normalization.
- Faithfulness and fidelity use perturbation baselines such as blur as part of the evaluation protocol.
- `run_pipeline.py` only covers summary aggregation and heatmap generation in this cleaned repo.
