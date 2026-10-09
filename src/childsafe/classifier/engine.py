"""Inference and metric summaries shared by train.py and evaluate.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from childsafe.classifier.calibration import binary_ece, ece, reliability_diagram, softmax


@torch.no_grad()
def run_inference(model, loader, device, amp: bool = True):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        targets.append(y)
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def summarize(logits: np.ndarray, targets: np.ndarray, classes: list[str], binary_map: dict,
              temperature: float = 1.0, reject_threshold: float | None = None,
              n_bins: int = 15, plot_prefix: str | Path | None = None, title: str = "") -> dict:
    """Per-class and binary (unsafe-vs-safe) metrics, before and after temperature scaling."""
    unsafe = np.array([binary_map[c] == "unsafe" for c in classes])
    res: dict = {"n": int(len(targets)), "temperature": temperature}
    present = sorted(set(targets.tolist()))
    for tag, t in (("raw", 1.0), ("calibrated", temperature)):
        p = softmax(logits, t)
        pred = p.argmax(1)
        conf = p.max(1)
        correct = (pred == targets).astype(float)
        p_unsafe = p[:, unsafe].sum(1)
        y_unsafe = unsafe[targets]
        res[f"ece_{tag}"] = ece(conf, correct, n_bins)
        res[f"binary_ece_{tag}"] = binary_ece(p_unsafe, y_unsafe, n_bins)
        res[f"nll_{tag}"] = float(-np.log(np.clip(p[np.arange(len(targets)), targets], 1e-12, 1)).mean())
        if plot_prefix is not None:
            reliability_diagram(conf, correct, f"{plot_prefix}_reliability_{tag}.png", n_bins, f"{title} ({tag})")

    p = softmax(logits, temperature)
    pred = p.argmax(1)
    res["accuracy"] = float(accuracy_score(targets, pred))
    res["macro_f1"] = float(f1_score(targets, pred, labels=present, average="macro", zero_division=0))
    prec, rec, f1, sup = precision_recall_fscore_support(targets, pred, labels=range(len(classes)), zero_division=0)
    res["per_class"] = {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                            "support": int(sup[i])} for i, c in enumerate(classes)}
    res["confusion_matrix"] = confusion_matrix(targets, pred, labels=range(len(classes))).tolist()

    # Binary view: P(unsafe) = sum of unsafe-category probabilities.
    p_unsafe = p[:, unsafe].sum(1)
    y_unsafe = unsafe[targets]
    b_pred = p_unsafe >= 0.5
    tp = int((b_pred & y_unsafe).sum()); fn = int((~b_pred & y_unsafe).sum())
    fp = int((b_pred & ~y_unsafe).sum()); tn = int((~b_pred & ~y_unsafe).sum())
    res["binary"] = {
        "accuracy": (tp + tn) / max(len(targets), 1),
        "unsafe_recall": tp / max(tp + fn, 1),
        "unsafe_precision": tp / max(tp + fp, 1),
        "unsafe_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }
    if reject_threshold is not None:
        # A3 reject option: low calibrated confidence -> "uncertain" (caution), not a hard label.
        b_conf = np.maximum(p_unsafe, 1 - p_unsafe)
        keep = (b_conf >= reject_threshold) & (p.max(1) >= reject_threshold)
        res["reject"] = {
            "threshold": reject_threshold,
            "coverage": float(keep.mean()),
            "binary_accuracy_on_kept": float((b_pred[keep] == y_unsafe[keep]).mean()) if keep.any() else float("nan"),
            "unsafe_missed_as_safe_confidently": int((~b_pred & y_unsafe & keep).sum()),
            "unsafe_rejected_to_caution": int((y_unsafe & ~keep).sum()),
        }
    return res
