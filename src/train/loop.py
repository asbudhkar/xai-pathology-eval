import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _to_1d_labels(y: torch.Tensor) -> np.ndarray:
    return y.detach().view(-1).cpu().numpy().astype(int)


def _safe_macro_ovr_auroc(y_true: np.ndarray, prob: np.ndarray) -> float | None:
    """
    y_true: [N] int labels in [0..K-1]
    prob:   [N,K] probability estimates (rows sum to 1)
    Returns macro AUROC (OVR) or None if not computable.
    """
    try:
        return float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except ValueError:
        return None


# Run one evaluation epoch.
def eval_epoch(
    model,
    dl,
    device,
    num_classes: int | None = None,
    compute_multiclass_auroc: bool = False,
):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()

    ys, ps = [], []
    prob1 = []   # binary: P(class=1)
    probK = []   # multiclass: softmax [B,K] if enabled

    total = 0.0
    n_seen = 0

    with torch.no_grad():
        for x, y in tqdm(dl, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.view(-1).long().to(device, non_blocking=True)

            logits = model(x)
            loss = ce(logits, y)

            bs = x.size(0)
            total += loss.item() * bs
            n_seen += bs

            pred = logits.argmax(dim=1)
            ys.append(_to_1d_labels(y))
            ps.append(_to_1d_labels(pred))

            k = num_classes if num_classes is not None else logits.shape[1]
            if k == 2:
                p = torch.softmax(logits, dim=1)[:, 1]
                prob1.append(p.detach().cpu().numpy())
            elif compute_multiclass_auroc:
                probK.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    ys = np.concatenate(ys) if ys else np.array([], dtype=int)
    ps = np.concatenate(ps) if ps else np.array([], dtype=int)

    loss_mean = total / max(1, n_seen)
    acc = accuracy_score(ys, ps) if ys.size else 0.0

    k = num_classes if num_classes is not None else (int(np.max(ys)) + 1 if ys.size else 2)

    if k == 2:
        f1 = f1_score(ys, ps, average="binary") if ys.size else 0.0
        auroc = None
        
        if prob1 and ys.size and len(np.unique(ys)) == 2:
            p1 = np.concatenate(prob1)
            auroc = float(roc_auc_score(ys, p1))
    else:
        f1 = f1_score(ys, ps, average="macro") if ys.size else 0.0
        auroc = None
        if compute_multiclass_auroc and probK and ys.size:
            P = np.concatenate(probK, axis=0)  # [N,K]
            auroc = _safe_macro_ovr_auroc(ys, P)

    return loss_mean, acc, f1, auroc


# Run one training epoch.
def train_epoch(
    model,
    dl,
    opt,
    device,
    num_classes: int | None = None,
    compute_multiclass_auroc: bool = False,
):
    model.train()
    ce = torch.nn.CrossEntropyLoss()

    ys, ps = [], []
    prob1 = []
    probK = []

    total = 0.0
    n_seen = 0

    for x, y in tqdm(dl, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.view(-1).long().to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()

        bs = x.size(0)
        total += loss.item() * bs
        n_seen += bs

        pred = logits.argmax(dim=1)
        ys.append(_to_1d_labels(y))
        ps.append(_to_1d_labels(pred))

        k = num_classes if num_classes is not None else logits.shape[1]
        if k == 2:
            p = torch.softmax(logits, dim=1)[:, 1]
            prob1.append(p.detach().cpu().numpy())
        elif compute_multiclass_auroc:
            probK.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    ys = np.concatenate(ys) if ys else np.array([], dtype=int)
    ps = np.concatenate(ps) if ps else np.array([], dtype=int)

    loss_mean = total / max(1, n_seen)
    acc = accuracy_score(ys, ps) if ys.size else 0.0

    k = num_classes if num_classes is not None else (int(np.max(ys)) + 1 if ys.size else 2)

    if k == 2:
        f1 = f1_score(ys, ps, average="binary") if ys.size else 0.0
        auroc = None
        if prob1 and ys.size and len(np.unique(ys)) == 2:
            p1 = np.concatenate(prob1)
            auroc = float(roc_auc_score(ys, p1))
    else:
        f1 = f1_score(ys, ps, average="macro") if ys.size else 0.0
        auroc = None
        if compute_multiclass_auroc and probK and ys.size:
            P = np.concatenate(probK, axis=0)
            auroc = _safe_macro_ovr_auroc(ys, P)

    return loss_mean, acc, f1, auroc
