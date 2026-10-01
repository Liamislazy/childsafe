"""Task 3 (A1): perceptual hashes -> near-duplicate groups -> group split and random split.

Writes data/splits/splits.csv (path, category, label, group, split_group, split_random),
docs/split_report.md and outputs/dedup/pairs_montage_*.jpg for manual threshold checks.

    python scripts/make_splits.py
    python scripts/make_splits.py --phash 8 --dhash 10   # override thresholds
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from childsafe.config import load_config
from childsafe.data.catalog import build_catalog
from childsafe.data.dedup import (compute_hashes, cross_split_leakage, find_duplicate_pairs,
                                  union_find_groups)
from childsafe.data.split import group_split, random_split


def montage(root, df, pairs, path, n=24, thumb=112):
    rng = np.random.default_rng(0)
    sel = pairs[rng.permutation(len(pairs))[:n]] if len(pairs) else pairs
    canvas = Image.new("RGB", (thumb * 4, thumb * ((len(sel) + 1) // 2)), "white")
    for k, (i, j, dp, dd) in enumerate(sel):
        for m, idx in enumerate((i, j)):
            im = Image.open(Path(root) / df.path.iloc[idx]).convert("RGB").resize((thumb, thumb))
            canvas.paste(im, ((k % 2) * 2 * thumb + m * thumb, (k // 2) * thumb))
        ImageDraw.Draw(canvas).text(((k % 2) * 2 * thumb + 2, (k // 2) * thumb + 2), f"p{dp} d{dd}", fill="yellow")
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phash", type=int)
    ap.add_argument("--dhash", type=int)
    ap.add_argument("--rehash", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, sc = cfg["dedup"], cfg["split"]
    tp, td = args.phash or dc["phash_threshold"], args.dhash or dc["dhash_threshold"]
    root = Path(cfg["paths"]["raw_data"])

    df = build_catalog(root, cfg["label_map"], cfg.get("category_aliases"))
    hash_file = Path(cfg["paths"]["hashes"])
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    if hash_file.exists() and not args.rehash and list(np.load(hash_file, allow_pickle=True)["paths"]) == list(df.path):
        z = np.load(hash_file, allow_pickle=True)
        ph, dh = z["phash"], z["dhash"]
    else:
        ph, dh = compute_hashes([root / p for p in df.path], dc["hash_size"], dc["dihedral"])
        np.savez(hash_file, phash=ph, dhash=dh, paths=np.array(df.path))

    report = ["# Split report (A1)\n", f"Images: {len(df)}. Dihedral hashing: {dc['dihedral']}.\n",
              "## Threshold sensitivity\n", "| pHash thr | dHash thr | dup pairs | groups | largest group | images in groups >1 |",
              "|---|---|---|---|---|---|"]
    for t in sorted({4, 6, 8, 10, 12, 14, tp}):
        pr = find_duplicate_pairs(ph, dh, t, t + (td - tp))
        g = union_find_groups(len(df), pr)
        sizes = np.bincount(g)
        report.append(f"| {t} | {t + td - tp} | {len(pr)} | {len(sizes)} | {sizes.max()} | {int(sizes[sizes > 1].sum())} |")

    pairs = find_duplicate_pairs(ph, dh, tp, td)
    # Filename evidence: augmented copies of one source photo (ChildSUn: <class>_<index>_0_<rand>.jpg).
    src_codes = df["source_id"].astype("category").cat.codes.to_numpy()
    src_pairs = np.array([(i, j, -1, -1) for idx in pd.Series(range(len(df))).groupby(src_codes).groups.values()
                          for i, j in zip(list(idx)[:-1], list(idx)[1:])], dtype=np.int64).reshape(-1, 4)
    hash_groups = union_find_groups(len(df), pairs)
    groups = union_find_groups(len(df), np.concatenate([pairs, src_pairs]))
    sizes = np.bincount(groups)
    n_src = df["source_id"].nunique()
    # how well does pHash alone recover the filename groups? (validates A1 for datasets without such names)
    multi = df.groupby("source_id").filter(lambda g: len(g) > 1)
    caught = sum(len(set(hash_groups[g.index])) == 1 for _, g in multi.groupby("source_id"))
    n_multi = multi["source_id"].nunique()
    report += [f"\n## Source photos from filenames\n",
               f"Distinct source photos: {n_src} (paper abstract: ~1,358 raw). "
               f"Sources with several augmented copies: {n_multi}.",
               f"pHash/dHash alone puts all copies of a source in one group for {caught}/{n_multi} sources "
               f"({caught / max(n_multi, 1):.1%}). Final groups = union of filename groups and hash groups.\n"]
    if sizes.max() > 0.05 * len(df):
        print(f"WARNING: largest group has {sizes.max()} images (>5%). Threshold may be too loose (chaining).")
    df["group"] = groups
    y = df["category"].to_numpy()
    df["split_group"] = group_split(y, groups, sc["test_frac"], sc["val_frac"], sc["seed"])
    df["split_random"] = random_split(y, sc["test_frac"], sc["val_frac"], sc["seed"])
    out_csv = Path(cfg["paths"]["splits"])
    df.to_csv(out_csv, index=False)
    import hashlib
    checksum = hashlib.md5("".join(f"{p}:{g}" for p, g in sorted(zip(df.path, df.split_group))).encode()).hexdigest()[:12]
    print(f"group split checksum: {checksum}  (identical on every machine = identical split)")

    report += [f"\n## Chosen thresholds: pHash <= {tp}, dHash <= {td}\n",
               f"Near-duplicate pairs: {len(pairs)}; groups: {len(sizes)}; largest: {sizes.max()}; "
               f"images sharing a group: {int(sizes[sizes > 1].sum())} ({sizes[sizes > 1].sum() / len(df):.1%}).\n",
               "## Split sizes\n", "| split | group split | random split |", "|---|---|---|"]
    for s in ("train", "val", "test"):
        report.append(f"| {s} | {(df.split_group == s).sum()} | {(df.split_random == s).sum()} |")
    report += ["\n## Cross-split near-duplicates (share of test images with a near-duplicate in train)\n",
               f"- random split: {cross_split_leakage(ph, dh, df.split_random.to_numpy(), tp, td):.1%}",
               f"- group split: {cross_split_leakage(ph, dh, df.split_group.to_numpy(), tp, td):.1%} (should be 0)\n",
               "Share of test images whose source photo (filename) also has a copy in train:\n",
               *[f"- {name} split: {df[df[col] == 'test'].source_id.isin(set(df[df[col] == 'train'].source_id)).mean():.1%}"
                 for name, col in (("random", "split_random"), ("group", "split_group"))],
               "",
               "Class balance of test split (group):\n", "```", df[df.split_group == 'test'].category.value_counts().to_string(), "```\n",
               "Check `outputs/dedup/pairs_montage_*.jpg` to confirm the pairs really are duplicates / augmentations."]
    mdir = Path(cfg["paths"]["outputs"]) / "dedup"
    mdir.mkdir(parents=True, exist_ok=True)
    montage(root, df, pairs, mdir / f"pairs_montage_p{tp}_d{td}.jpg")
    # borderline pairs just above threshold: are we missing real duplicates?
    near = find_duplicate_pairs(ph, dh, tp + 4, td + 4)
    border = near[(near[:, 2] > tp) | (near[:, 3] > td)]
    montage(root, df, border, mdir / f"pairs_montage_borderline.jpg")

    Path(cfg["paths"]["docs"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["docs"]) / "split_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
