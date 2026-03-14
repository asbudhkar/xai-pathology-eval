import argparse, json, re, math
from pathlib import Path
import pandas as pd
import numpy as np


def read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def safe_read_json(p: Path):
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception:
        return None

def safe_read_csv(p: Path):
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def read_sanity_csv(p: Path):
    """
    expects header: stage,mean,std,count
    returns dict of selected stages
    """
    df = safe_read_csv(p)
    if df is None or df.empty:
        return {}

    keep = [
        "original",
        "rand_head",
        "rand_last25",
        "rand_last50",
        "rand_all",
        "rand_conv1",
        "rand_layer4",
        "rand_fc",
    ]
    df = df[df["stage"].isin(keep)].copy()
    out = {}
    for _, r in df.iterrows():
        stage = str(r["stage"])
        out[f"sanity_{stage}_mean"] = float(r["mean"])
        out[f"sanity_{stage}_std"]  = float(r["std"])
        out[f"sanity_{stage}_count"]= int(r["count"]) if "count" in r else np.nan
    return out

def read_sanity_meta(p: Path):
    m = safe_read_json(p)
    if m is None:
        return {}
    return {
        "sanity_time_total_sec": float(m.get("time_total_sec", np.nan)),
        "sanity_time_per_image_sec": float(m.get("time_per_image_sec", np.nan)),
    }

def parse_run_name(run_name: str):
    """
    Parse names like:
      pathmnist_resnet18_seed0
      pcam_resnet18_seed0
    """
    base = run_name

    m = re.match(r"(?P<dataset>.+)_(?P<model>resnet18|efficientnet_b0|uni|virchow)_seed(?P<seed>\d+)$", base)
    if m:
        d = m.groupdict()
        d["seed"] = int(d["seed"])
        d["condition"] = "trained"
        return d

    return {"dataset": None, "model": None, "seed": None, "condition": "trained"}

def read_train_metrics(runs_root: Path, run_name: str):
    """
    Reads outputs/runs/<run_name>/metrics.json (if present).
    Returns dict of a few key fields.
    """
    mpath = runs_root / run_name / "metrics.json"
    m = safe_read_json(mpath)
    if m is None:
        return {}

    out = {}
    out["train_best_val_acc"] = float(m.get("best_val_acc", np.nan))
    out["train_num_classes"] = int(m.get("num_classes", m.get("num_classes", np.nan))) if m.get("num_classes", None) is not None else np.nan

    test = m.get("test", {})
    out["test_loss"]  = float(test.get("loss", np.nan))
    out["test_acc"]   = float(test.get("acc", np.nan))
    out["test_f1"]    = float(test.get("f1", np.nan))
    out["test_auroc"] = float(test.get("auroc", np.nan)) if test.get("auroc", None) is not None else np.nan
    return out

# Add faithfulness metrics
def add_faithfulness(row: dict, f: dict, col_suffix: str):
    # Handle the f helper step.
    def _f(v):
        return float(v) if v is not None else np.nan

    keys = [
        "faith_del_mean",
        "faith_del_std",
        "faith_ins_mean",
        "faith_ins_std",
        "faith_gap_mean",
        "faith_gap_std",
        "faith_del_acc_auc_mean",
        "faith_del_acc_auc_std",
        "faith_ins_acc_auc_mean",
        "faith_ins_acc_auc_std",
        "faith_acc_gap_mean",
        "faith_acc_gap_std",
        "del_pct",
        "del_base_acc",
        "del_top_acc",
        "del_rand_acc",
        "del_acc_drop_top",
        "del_acc_drop_rand",
        "del_base_auroc",
        "del_top_auroc",
        "del_rand_auroc",
        "del_auroc_drop_top",
        "del_auroc_drop_rand",
        "faith_time_total_sec",
        "faith_time_per_sample_sec",
    ]

    if f is not None:
        row[f"faith_del_mean{col_suffix}"] = float(f.get("deletion_auc_mean", np.nan))
        row[f"faith_del_std{col_suffix}"]  = float(f.get("deletion_auc_std", np.nan))
        row[f"faith_ins_mean{col_suffix}"] = float(f.get("insertion_auc_mean", np.nan))
        row[f"faith_ins_std{col_suffix}"]  = float(f.get("insertion_auc_std", np.nan))
        row[f"faith_gap_mean{col_suffix}"] = row[f"faith_ins_mean{col_suffix}"] - row[f"faith_del_mean{col_suffix}"]
        row[f"faith_del_acc_auc_mean{col_suffix}"] = float(f.get("deletion_acc_auc_mean", np.nan))
        row[f"faith_del_acc_auc_std{col_suffix}"] = float(f.get("deletion_acc_auc_std", np.nan))
        row[f"faith_ins_acc_auc_mean{col_suffix}"] = float(f.get("insertion_acc_auc_mean", np.nan))
        row[f"faith_ins_acc_auc_std{col_suffix}"] = float(f.get("insertion_acc_auc_std", np.nan))
        row[f"faith_acc_gap_mean{col_suffix}"] = (
            row[f"faith_ins_acc_auc_mean{col_suffix}"] - row[f"faith_del_acc_auc_mean{col_suffix}"]
        )
        row[f"del_pct{col_suffix}"] = float(f.get("del_pct", np.nan))
        row[f"del_base_acc{col_suffix}"] = float(f.get("base_acc", np.nan))
        row[f"del_top_acc{col_suffix}"] = float(f.get("del_top_acc", np.nan))
        row[f"del_rand_acc{col_suffix}"] = float(f.get("del_rand_acc", np.nan))
        row[f"del_acc_drop_top{col_suffix}"] = float(f.get("del_acc_drop_top", np.nan))
        row[f"del_acc_drop_rand{col_suffix}"] = float(f.get("del_acc_drop_rand", np.nan))
        row[f"del_base_auroc{col_suffix}"] = _f(f.get("auroc_base", np.nan))
        row[f"del_top_auroc{col_suffix}"] = _f(f.get("auroc_del_top", np.nan))
        row[f"del_rand_auroc{col_suffix}"] = _f(f.get("auroc_del_rand", np.nan))
        row[f"del_auroc_drop_top{col_suffix}"] = _f(f.get("auroc_drop_top", np.nan))
        row[f"del_auroc_drop_rand{col_suffix}"] = _f(f.get("auroc_drop_rand", np.nan))
        row[f"faith_time_total_sec{col_suffix}"] = float(f.get("time_total_sec", np.nan))
        row[f"faith_time_per_sample_sec{col_suffix}"] = float(f.get("time_per_sample_sec", np.nan))

        if np.isfinite(row[f"faith_ins_std{col_suffix}"]) and np.isfinite(row[f"faith_del_std{col_suffix}"]):
            row[f"faith_gap_std{col_suffix}"] = float(math.sqrt(
                row[f"faith_ins_std{col_suffix}"]**2 + row[f"faith_del_std{col_suffix}"]**2
            ))
        else:
            row[f"faith_gap_std{col_suffix}"] = np.nan
        if np.isfinite(row[f"faith_ins_acc_auc_std{col_suffix}"]) and np.isfinite(row[f"faith_del_acc_auc_std{col_suffix}"]):
            row[f"faith_acc_gap_std{col_suffix}"] = float(math.sqrt(
                row[f"faith_ins_acc_auc_std{col_suffix}"]**2 + row[f"faith_del_acc_auc_std{col_suffix}"]**2
            ))
        else:
            row[f"faith_acc_gap_std{col_suffix}"] = np.nan
    else:
        for key in keys:
            row[f"{key}{col_suffix}"] = np.nan


# Add fidelity metrics
def add_fidelity(row: dict, f: dict):
    if f is None:
        row["fid_corr_mean"] = np.nan
        row["fid_corr_std"] = np.nan
        row["fid_corr_count"] = np.nan
        row["fid_time_total_sec"] = np.nan
        row["fid_time_per_sample_sec"] = np.nan
        return
    row["fid_corr_mean"] = float(f.get("fidelity_corr_mean", np.nan))
    row["fid_corr_std"] = float(f.get("fidelity_corr_std", np.nan))
    row["fid_corr_count"] = float(f.get("fidelity_corr_count", np.nan))
    row["fid_time_total_sec"] = float(f.get("time_total_sec", np.nan))
    row["fid_time_per_sample_sec"] = float(f.get("time_per_sample_sec", np.nan))


# Collect all saved metrics for one run and explainer.
def one_run(results_dir: Path, explainer: str, runs_root: Path, faith_fills: list[str]):
    meta = parse_run_name(results_dir.name)
    row = {
        "run": results_dir.name,
        "dataset": meta["dataset"],
        "model": meta["model"],
        "seed": meta["seed"],
        "condition": meta["condition"],
        "explainer": explainer,
    }

    # add training/test metrics
    row.update(read_train_metrics(runs_root, results_dir.name))

    fpath = results_dir / f"faithfulness_{explainer}_summary.json"
    rpath = results_dir / f"robustness_{explainer}_summary.json"
    fipath = results_dir / f"fidelity_{explainer}_summary.json"
    spath = results_dir / f"sanity_weights_{explainer}_summary.csv"
    smeta = results_dir / f"sanity_weights_{explainer}_summary.json"

    f = safe_read_json(fpath)
    r = safe_read_json(rpath)
    fi = safe_read_json(fipath)
    s = read_sanity_csv(spath)
    sm = read_sanity_meta(smeta)

    # faithfulness (default + optional fills)
    add_faithfulness(row, f, "")
    for fill in faith_fills:
        fill_suffix = f"_{fill}"
        f_fill_path = results_dir / f"faithfulness_{explainer}{fill_suffix}_summary.json"
        f_fill = safe_read_json(f_fill_path)
        # If blur-specific files are missing but the base run used blur, mirror it into blur columns.
        if f_fill is None and fill == "blur" and f is not None and f.get("fill") == "blur":
            f_fill = f
            row[f"has_faith{fill_suffix}"] = True
        else:
            row[f"has_faith{fill_suffix}"] = f_fill_path.exists()
        add_faithfulness(row, f_fill, fill_suffix)

    # fidelity
    add_fidelity(row, fi)

    # robustness
    if r is not None:
        row["rob_hflip_mean"]  = float(r.get("hflip_mean", np.nan))
        row["rob_hflip_std"]   = float(r.get("hflip_std", np.nan))
        row["rob_vflip_mean"]  = float(r.get("vflip_mean", np.nan))
        row["rob_vflip_std"]   = float(r.get("vflip_std", np.nan))
        row["rob_jitter_mean"] = float(r.get("jitter_mean", np.nan))
        row["rob_jitter_std"]  = float(r.get("jitter_std", np.nan))
        row["rob_hflip_samepred_mean"] = float(r.get("hflip_samepred_mean", np.nan))
        row["rob_hflip_samepred_count"] = float(r.get("hflip_samepred_count", np.nan))
        row["rob_vflip_samepred_mean"] = float(r.get("vflip_samepred_mean", np.nan))
        row["rob_vflip_samepred_count"] = float(r.get("vflip_samepred_count", np.nan))
        row["rob_jitter_samepred_mean"] = float(r.get("jitter_samepred_mean", np.nan))
        row["rob_jitter_samepred_count"] = float(r.get("jitter_samepred_count", np.nan))
        row["rob_time_total_sec"] = float(r.get("time_total_sec", np.nan))
        row["rob_time_per_sample_sec"] = float(r.get("time_per_sample_sec", np.nan))
    else:
        row["rob_hflip_mean"]  = np.nan
        row["rob_hflip_std"]   = np.nan
        row["rob_vflip_mean"]  = np.nan
        row["rob_vflip_std"]   = np.nan
        row["rob_jitter_mean"] = np.nan
        row["rob_jitter_std"]  = np.nan
        row["rob_time_total_sec"] = np.nan
        row["rob_time_per_sample_sec"] = np.nan

    # sanity
    row.update(s)
    row.update(sm)

    # existence flags
    row["has_faith"] = fpath.exists()
    row["has_fid"] = fipath.exists()
    row["has_rob"]   = rpath.exists()
    row["has_sanity"]= spath.exists()
    row["has_train_metrics"] = (runs_root / results_dir.name / "metrics.json").exists()
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="./outputs/results",
                    help="Parent folder that contains run dirs with *_summary.json files")
    ap.add_argument("--runs_root", default="./outputs/runs",
                    help="Parent folder that contains run dirs with metrics.json + ckpt_best.pt")
    ap.add_argument("--explainers", default="ig",
                    help="Comma-separated explainers to summarize (must match filenames)")
    ap.add_argument("--faith_fills", default="blur,mean",
                    help="Comma-separated fill modes to read, e.g. blur,mean (matches out_suffix)")
    ap.add_argument("--out_csv", default="./outputs/summary/summary_all.csv")
    ap.add_argument("--per_dataset_dir", default="./outputs/summary",
                    help="Also writes <dataset>_xai_summary.csv into this folder")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    runs_root = Path(args.runs_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    expl = [e.strip() for e in args.explainers.split(",") if e.strip()]
    faith_fills = [f.strip() for f in args.faith_fills.split(",") if f.strip()]

    run_dirs = sorted([d for d in results_root.iterdir() if d.is_dir() and not d.name.endswith("_labelshuffle")])

    rows = []
    for d in run_dirs:
        for e in expl:
            rows.append(one_run(d, e, runs_root, faith_fills))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    per_dir = Path(args.per_dataset_dir)
    per_dir.mkdir(parents=True, exist_ok=True)
    for ds, g in df.groupby("dataset", dropna=True):
        g.to_csv(per_dir / f"{ds}_xai_summary.csv", index=False)

    cols = [
        "dataset","model","seed","condition","explainer",
        "test_acc","test_f1","test_auroc",
        "faith_gap_mean","rob_jitter_mean","rob_jitter_samepred_mean",
        "del_acc_drop_top","del_acc_drop_rand","del_auroc_drop_top",
        "fid_corr_mean",
        "faith_time_total_sec","rob_time_total_sec","sanity_time_total_sec",
        "sanity_rand_last25_mean","sanity_rand_head_mean",
        "has_faith","has_fid","has_rob","has_sanity","has_train_metrics"
    ]
    cols = [c for c in cols if c in df.columns]
    if len(df):
        print(df[cols].sort_values(["dataset","explainer"]).to_string(index=False))
    else:
        print("[WARN] No rows produced. Check outputs/results structure + explainer names.")

    print("\nSaved:", out_csv.resolve())
    print("Per-dataset CSVs in:", per_dir.resolve())

if __name__ == "__main__":
    main()

# python scripts/make_summary_all.py \
#   --results_root outputs/results \
#   --runs_root outputs/runs \
#   --explainers ig,gradcam \
#   --out_csv outputs/summary/summary_all.csv
