"""Task 2 of the handoff: record folder structure, class counts, image sizes and signs of
pre-applied augmentation. Writes docs/dataset_report.md.

    python scripts/inspect_dataset.py
"""
import argparse
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from childsafe.config import load_config
from childsafe.data.catalog import IMG_EXT, build_catalog

AUG_HINT = re.compile(r"(aug|augment|_rot|_flip|copy|\(\d+\)|_\d+_\d+)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    root = Path(cfg["paths"]["raw_data"])
    lines = [f"# ChildSUn dataset report\n", f"Root: `{root}`\n", "## Folder tree (image counts)\n", "```"]
    counts = Counter()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            counts[p.parent.relative_to(root).as_posix()] += 1
    for k in sorted(counts):
        lines.append(f"{k}: {counts[k]}")
    lines.append("```\n")
    other = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file() and p.suffix.lower() not in IMG_EXT)
    if other:
        lines.append(f"Non-image files by extension: {dict(other)}\n")

    df = build_catalog(root, cfg["label_map"], cfg.get("category_aliases"))
    lines.append(f"## Catalog\n\nLabelled images: **{len(df)}**\n")
    lines.append("| category | binary | count |\n|---|---|---|")
    for (c, b), n in df.groupby(["category", "label"]).size().items():
        lines.append(f"| {c} | {b} | {n} |")
    lines.append(f"\nBinary totals: {df['label'].value_counts().to_dict()}\n")
    if df["source_split"].str.len().gt(0).any():
        lines.append(f"Dataset ships its own split folders: {df['source_split'].value_counts().to_dict()} "
                     "(not used; see A1).\n")

    sizes = Counter()
    for p in df["path"].sample(min(500, len(df)), random_state=0):
        try:
            with Image.open(root / p) as im:
                sizes[im.size] += 1
        except Exception:
            sizes["unreadable"] += 1
    lines.append(f"Image sizes (sample of {sum(sizes.values())}): {dict(sizes.most_common(8))}\n")
    hinted = df["path"].map(lambda s: bool(AUG_HINT.search(Path(s).stem)))
    lines.append(f"File names matching augmentation-like patterns: {int(hinted.sum())} "
                 f"(e.g. {list(df.loc[hinted, 'path'].head(5))})\n")
    lines.append("\n## Still to check by hand\n\n- License (see dataset-metadata.json from the download script)\n"
                 "- Whether augmented copies are included (compare with the ~1,358 raw / ~5,350 total in the paper abstract; "
                 "see near-duplicate groups from scripts/make_splits.py)\n- Safe/unsafe grouping of each category vs the paper\n")
    out = Path(cfg["paths"]["docs"]) / "dataset_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
