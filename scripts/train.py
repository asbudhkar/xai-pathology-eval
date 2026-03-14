import os, json, argparse
import torch

from src.utils.seed import set_seed, setup_cuda_speed
from src.datasets.pathmnist import get_pathmnist_loaders
from src.datasets.bloodmnist import get_bloodmnist_loaders
from src.datasets.pcam import get_pcam_loaders
from src.models.cnn import make_resnet18, make_efficientnet_b0, make_uni, make_virchow
from src.train.loop import train_epoch, eval_epoch

def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--data_root", type=str, default="datasets")
    ap.add_argument("--download", action="store_true", help="download dataset if missing")
    ap.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18", "efficientnet_b0", "uni", "virchow"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dataset", type=str, default="pathmnist",
                choices=["pathmnist", "bloodmnist", "pcam"])
    return ap.parse_args()

def main():
    args = parse_args()
    setup_cuda_speed()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA not available—check your env."

    if args.dataset == "pathmnist":
        train_dl, val_dl, test_dl, num_classes = get_pathmnist_loaders(
            img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif args.dataset == "bloodmnist":
        train_dl, val_dl, test_dl, num_classes = get_bloodmnist_loaders(
            img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif args.dataset == "pcam":
        train_dl, val_dl, test_dl, num_classes = get_pcam_loaders(
            img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers, root="./pcam"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.model == "resnet18":
        model = make_resnet18(num_classes, pretrained=True)
    elif args.model == "efficientnet_b0":
        model = make_efficientnet_b0(num_classes, pretrained=True)
    elif args.model == "uni":
        model = make_uni(num_classes, pretrained=True, freeze_backbone=True)
    elif args.model == "virchow":
        model = make_virchow(num_classes, pretrained=True, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    outdir = f"./outputs/runs/{args.dataset}_{args.model}_seed{args.seed}"
    os.makedirs(outdir, exist_ok=True)

    best_val = -1.0
    best_path = os.path.join(outdir, "ckpt_best.pt")

    history = []
    for ep in range(args.epochs):
        tr = train_epoch(model, train_dl, opt, device, num_classes=num_classes, compute_multiclass_auroc=True)
        va = eval_epoch(model, val_dl, device, num_classes=num_classes, compute_multiclass_auroc=True)
        history.append({"epoch": ep,
            "train": {"loss": tr[0], "acc": tr[1], "f1": tr[2], "auroc": tr[3]},
            "val":   {"loss": va[0], "acc": va[1], "f1": va[2], "auroc": va[3]},
        })
        msg = (f"ep {ep} | train loss {tr[0]:.4f} acc {tr[1]:.4f} f1 {tr[2]:.4f}"
            f" | val loss {va[0]:.4f} acc {va[1]:.4f} f1 {va[2]:.4f}")
        if tr[3] is not None and va[3] is not None:
            msg += f" auroc {va[3]:.4f}"
        print(msg)

        if va[1] > best_val:
            best_val = va[1]
            torch.save({"model": model.state_dict(), "args": vars(args), "num_classes": num_classes}, best_path)

    # final test with best checkpoint
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    te = eval_epoch(model, test_dl, device, num_classes=num_classes, compute_multiclass_auroc=True)

    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "num_classes": num_classes,
        "best_val_acc": best_val,
        "test": {"loss": te[0], "acc": te[1], "f1": te[2], "auroc": te[3]},
        "history": history,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", best_path)
    print("Test:", metrics["test"])

if __name__ == "__main__":
    main()
