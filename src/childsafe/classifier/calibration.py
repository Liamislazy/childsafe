"""A3: temperature scaling, ECE, reliability diagrams (Guo et al., ICML 2017 method)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, targets: np.ndarray, max_iter: int = 200) -> float:
    """Fit a single temperature T > 0 minimising NLL on held-out (val) logits."""
    lg = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def binary_ece(p_pos: np.ndarray, y_pos: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the binary (unsafe) prediction, using confidence of the predicted side."""
    pred = p_pos >= 0.5
    conf = np.where(pred, p_pos, 1 - p_pos)
    return ece(conf, (pred == y_pos).astype(float), n_bins)


def reliability_diagram(confidence, correct, path: str | Path, n_bins: int = 15, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc, cnt = np.zeros(n_bins), np.zeros(n_bins)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (confidence > lo) & (confidence <= hi)
        cnt[b] = m.sum()
        acc[b] = correct[m].mean() if m.any() else np.nan
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.5, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.bar(centers, np.nan_to_num(acc), width=1 / n_bins, edgecolor="black", alpha=0.8)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE = {ece(confidence, correct, n_bins):.3f}")
    ax2.bar(centers, cnt, width=1 / n_bins, color="gray")
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
