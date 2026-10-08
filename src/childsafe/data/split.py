"""A1: group-based split vs plain random split.

Both splits use the same train/val/test fractions and stratify by category, so the
only difference is whether near-duplicate groups are kept together. The gap in test
accuracy between the two is our estimate of how much leakage inflates the score.

The group split uses its own greedy assignment with numpy's PCG64 generator instead of
sklearn's StratifiedGroupKFold: the sklearn result changed between library versions
(local vs Kaggle gave different splits from identical groups), which breaks reproducibility.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

PARTS = ("train", "val", "test")


def _strata(y: np.ndarray, min_count: int = 3) -> np.ndarray:
    """Stratification labels; classes too small to stratify are pooled (sklearn needs >= 2 per class)."""
    y = np.asarray(y, dtype=object)
    vals, counts = np.unique(y, return_counts=True)
    rare = set(vals[counts < min_count])
    if not rare:
        return y
    s = np.array(["__rare__" if v in rare else v for v in y], dtype=object)
    if (s == "__rare__").sum() < min_count:
        s[s == "__rare__"] = vals[counts.argmax()]
    return s


def random_split(y: np.ndarray, test_frac: float, val_frac: float, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=test_frac, stratify=_strata(y), random_state=seed)
    train, val = train_test_split(trval, test_size=val_frac / (1 - test_frac),
                                  stratify=_strata(np.asarray(y)[trval]), random_state=seed)
    out = np.empty(len(y), dtype=object)
    out[train], out[val], out[test] = "train", "val", "test"
    return out


def group_split(y: np.ndarray, groups: np.ndarray, test_frac: float, val_frac: float,
                seed: int) -> np.ndarray:
    """Greedy stratified group split: groups are visited largest first (random order within a
    size), and each goes to the part that still needs the most images of the group's classes."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes, y_idx = np.unique(y, return_inverse=True)
    gids, g_idx = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(gids), len(classes)), dtype=np.int64)
    np.add.at(counts, (g_idx, y_idx), 1)

    frac = np.array([1 - test_frac - val_frac, val_frac, test_frac])
    target = frac[:, None] * counts.sum(0)[None, :]          # (3, n_classes)
    have = np.zeros_like(target)
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(len(gids))
    order = order[np.argsort(-counts[order].sum(1), kind="stable")]

    assign = np.empty(len(gids), dtype=np.int64)
    for g in order:
        need = ((target - have) * counts[g][None, :]).sum(1) / counts[g].sum()
        part = int(np.argmax(need))
        assign[g] = part
        have[part] += counts[g]

    out = np.array(PARTS, dtype=object)[assign[g_idx]]
    for a in ("val", "test"):
        assert not (set(groups[out == "train"]) & set(groups[out == a])), f"group leaked into {a}"
    return out
