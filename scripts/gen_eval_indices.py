#!/usr/bin/env python3
import os
import json
import argparse
import random
import hashlib
import numpy as np
import pandas as pd

from src.datasets.pathmnist import get_pathmnist_loaders
from src.datasets.bloodmnist import get_bloodmnist_loaders
from src.datasets.pcam import get_pcam_loaders


# Group dataset indices by class label.
def stratified_indices_from_dataset(ds, num_classes):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["pathmnist", "bloodmnist", "pcam"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--outdir", default="./outputs/indices")
    ap.add_argument("--pcam_root", default="./pcam")
    ap.add_argument("--pcam_download", action="store_true")
    ap.add_argument("--indices_path", default=None,
                    help="Override output CSV path")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "pathmnist":
        _, _, test_dl, _ = get_pathmnist_loaders(img_size=args.img_size, batch_size=1, num_workers=0)
    elif args.dataset == "bloodmnist":
        _, _, test_dl, _ = get_bloodmnist_loaders(img_size=args.img_size, batch_size=1, num_workers=0)
    else:
        _, _, test_dl, _ = get_pcam_loaders(
            root=args.pcam_root,
            img_size=args.img_size,
            batch_size=1,
            num_workers=0,
            download=bool(args.pcam_download),
        )

    test_ds = test_dl.dataset
    buckets = stratified_indices_from_dataset(test_ds, num_classes=None)
    n_classes = len(buckets) if len(buckets) > 0 else 1

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

    if args.indices_path:
        out_path = args.indices_path
    else:
        out_path = os.path.join(args.outdir, f"eval_indices_{args.dataset}_normal.csv")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame({"idx": [int(v) for v in chosen]}).to_csv(out_path, index=False)

    hash8 = hashlib.sha1(",".join(str(i) for i in chosen).encode("utf-8")).hexdigest()[:8]
    print(f"[INDICES] n={len(chosen)} path={out_path} sha1={hash8}")


if __name__ == "__main__":
    main()
