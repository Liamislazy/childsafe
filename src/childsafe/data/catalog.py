"""Build an image catalog (path, category, binary label) from a folder tree.

The ChildSUn folder layout was not verified before writing this, so the category is
inferred from folder names: the deepest folder whose normalized name is in
``label_map`` wins, and a specific category (``knife``) beats a generic one (``unsafe``).
Folders called train/test/val are kept as ``source_split`` for information only;
they are NOT used for our split because they may contain leaked augmentations.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from childsafe.config import normalize_name

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "training", "test", "testing", "val", "valid", "validation"}
GENERIC = {"safe", "unsafe"}


def infer_labels(rel_parts: tuple[str, ...], label_map: dict) -> tuple[str | None, str | None, str | None]:
    """Return (category, binary_label, source_split) from the folder parts of a relative path."""
    category = binary = generic = source_split = None
    for part in rel_parts:
        norm = normalize_name(part)
        if norm in SPLIT_NAMES:
            source_split = norm
        if norm in label_map:
            if norm in GENERIC:
                generic = norm
            else:
                category, binary = norm, label_map[norm]
    if category is None and generic is not None:
        category, binary = generic, generic
    return category, binary, source_split


AUG_NAME = re.compile(r"^(?P<prefix>.+?)_(?P<src>\d+)_\d+_\d+$")   # Keras save_to_dir: <prefix>_<index>_<n>_<rand>
PLAIN_NAME = re.compile(r"^(?P<prefix>.+?)_\d+$")                   # e.g. fruits_12


def parse_filename(stem: str) -> tuple[str | None, str | None]:
    """Return (class prefix, source photo index or None)."""
    m = AUG_NAME.match(stem)
    if m:
        return m["prefix"], m["src"]
    m = PLAIN_NAME.match(stem)
    return (m["prefix"], None) if m else (None, None)


def build_catalog(root: str | Path, label_map: dict, aliases: dict | None = None) -> pd.DataFrame:
    root = Path(root)
    aliases = aliases or {}
    rows, unmatched = [], {}
    # sort by the posix string: Path sorting is case-insensitive on Windows, case-sensitive on Linux,
    # which gave different image orders (and different group splits) locally and on Kaggle
    for p in sorted(root.rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        rel = p.relative_to(root)
        category, binary, source_split = infer_labels(rel.parts[:-1], label_map)
        prefix, source = parse_filename(p.stem)
        if category is None or category in GENERIC:
            # flat layout (as ChildSUn ships): class is the filename prefix
            norm = normalize_name(prefix) if prefix else ""
            if norm in label_map and norm not in GENERIC:
                category, binary = norm, label_map[norm]
        if category is None:
            key = f"{rel.parent} ({prefix or p.stem})"
            unmatched[key] = unmatched.get(key, 0) + 1
            continue
        category = aliases.get(category, category)
        rows.append({"path": rel.as_posix(), "category": category, "label": binary,
                     "source_split": source_split or "",
                     # augmented copies of one original photo share this id
                     "source_id": f"{normalize_name(prefix)}_{source}" if source is not None else rel.as_posix()})
    if unmatched:
        print(f"[catalog] {sum(unmatched.values())} images not matched to label_map (skipped):")
        for k, v in sorted(unmatched.items()):
            print(f"    {k}: {v}")
    if not rows:
        raise SystemExit(f"No labelled images found under {root}. Check paths.raw_data and label_map.")
    return pd.DataFrame(rows)
