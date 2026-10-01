"""A1: perceptual-hash near-duplicate grouping.

Each image gets a 64-bit pHash and dHash. With ``dihedral=True`` we also hash the 7
flipped/rotated variants, and the distance between images i and j is the minimum over
j's variants. That catches copies produced by flip / 90-degree rotation augmentation,
which a plain pHash misses. Small-angle rotations, crops and colour jitter mostly keep
pHash close anyway. Two images are near-duplicates when BOTH hash distances are under
threshold; near-duplicate pairs are merged into groups with union-find.

Hash thresholds are a judgement call: inspect the montage from scripts/find_duplicates.py
at the chosen threshold and report the threshold in the write-up.
"""
from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

N_VARIANTS = 8


def _dihedral(img: Image.Image) -> list[Image.Image]:
    out = []
    for flip in (False, True):
        base = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else img
        for k in range(4):
            out.append(base.rotate(90 * k, expand=True) if k else base)
    return out


def _to_int(h: imagehash.ImageHash) -> np.uint64:
    bits = np.packbits(h.hash.flatten().astype(np.uint8))
    return np.frombuffer(bits.tobytes(), dtype=">u8")[0].astype(np.uint64)


def compute_hashes(paths: list[str | Path], hash_size: int = 8, dihedral: bool = True):
    """Return (phash, dhash) arrays of shape (N, 8) uint64. Unreadable images get all-ones hashes."""
    if hash_size != 8:
        raise ValueError("only 64-bit hashes (hash_size=8) are supported")
    n = len(paths)
    ph = np.full((n, N_VARIANTS), np.iinfo(np.uint64).max, dtype=np.uint64)
    dh = ph.copy()
    bad = []
    for i, p in enumerate(tqdm(paths, desc="hashing")):
        try:
            img = Image.open(p).convert("L")
        except Exception:  # corrupt file: keep sentinel hash so it matches nothing
            bad.append(str(p))
            continue
        variants = _dihedral(img) if dihedral else [img] * N_VARIANTS
        for v, im in enumerate(variants):
            ph[i, v] = _to_int(imagehash.phash(im, hash_size))
            dh[i, v] = _to_int(imagehash.dhash(im, hash_size))
    if bad:
        print(f"[dedup] {len(bad)} unreadable images, e.g. {bad[:3]}")
    return ph, dh


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1)


def min_distance_matrix_block(h: np.ndarray, rows: slice, cols: np.ndarray | None = None) -> np.ndarray:
    """Min Hamming distance between canonical hashes h[rows, 0] and all variants of h[cols]."""
    a = h[rows, 0][:, None, None]
    b = h[cols] if cols is not None else h
    return _popcount(a ^ b[None, :, :]).min(-1)


def find_duplicate_pairs(ph, dh, phash_thr: int, dhash_thr: int, chunk: int = 256):
    """Return array of (i, j, d_phash, d_dhash) for i < j that are near-duplicates."""
    n = len(ph)
    out = []
    for s in tqdm(range(0, n, chunk), desc="pairwise"):
        rows = slice(s, min(s + chunk, n))
        dp = min_distance_matrix_block(ph, rows)
        dd = min_distance_matrix_block(dh, rows)
        ii, jj = np.nonzero((dp <= phash_thr) & (dd <= dhash_thr))
        ii = ii + s
        keep = ii != jj
        # the variant-min distance is not symmetric, so keep a match found in either direction
        for i, j in zip(ii[keep], jj[keep]):
            out.append((min(i, j), max(i, j), dp[i - s, j], dd[i - s, j]))
    if not out:
        return np.zeros((0, 4), dtype=np.int64)
    arr = np.array(out, dtype=np.int64)
    arr = arr[np.lexsort((arr[:, 2] + arr[:, 3], arr[:, 1], arr[:, 0]))]
    _, first = np.unique(arr[:, :2], axis=0, return_index=True)
    return arr[np.sort(first)]


def union_find_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs[:, :2]:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group


def cross_split_leakage(ph, dh, split: np.ndarray, phash_thr: int, dhash_thr: int,
                        query: str = "test", ref: str = "train") -> float:
    """Fraction of `query` images that have a near-duplicate in `ref`."""
    q = np.nonzero(split == query)[0]
    r = np.nonzero(split == ref)[0]
    if len(q) == 0 or len(r) == 0:
        return float("nan")
    hits = 0
    for s in range(0, len(q), 256):
        idx = q[s:s + 256]
        dp = _popcount(ph[idx, 0][:, None, None] ^ ph[r][None]).min(-1)
        dd = _popcount(dh[idx, 0][:, None, None] ^ dh[r][None]).min(-1)
        hits += int(((dp <= phash_thr) & (dd <= dhash_thr)).any(1).sum())
    return hits / len(q)
