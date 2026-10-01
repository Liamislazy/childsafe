import numpy as np
import pandas as pd
from PIL import Image

from childsafe.data.catalog import build_catalog, infer_labels
from childsafe.data.corruptions import KINDS, corrupt
from childsafe.data.dedup import compute_hashes, cross_split_leakage, find_duplicate_pairs, union_find_groups
from childsafe.data.split import group_split, random_split

LABELS = {"knife": "unsafe", "toy": "safe", "safe": "safe", "unsafe": "unsafe"}


def test_infer_labels_prefers_specific_category():
    assert infer_labels(("train", "unsafe", "Knife"), LABELS) == ("knife", "unsafe", "train")
    assert infer_labels(("safe",), LABELS) == ("safe", "safe", None)
    assert infer_labels(("other",), LABELS) == (None, None, None)


def _img(seed, size=96):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)


def test_dedup_finds_flipped_and_rotated_copies(tmp_path):
    paths = []
    for k in range(6):
        im = _img(k)
        p = tmp_path / f"{k}.png"; im.save(p); paths.append(p)
    # augmented copies of image 0: flip, rotate 90, brightness change
    extra = [_img(0).transpose(Image.Transpose.FLIP_LEFT_RIGHT), _img(0).rotate(90),
             Image.fromarray(np.clip(np.asarray(_img(0)).astype(int) + 20, 0, 255).astype(np.uint8))]
    for k, im in enumerate(extra):
        p = tmp_path / f"aug{k}.png"; im.save(p); paths.append(p)
    ph, dh = compute_hashes(paths)
    pairs = find_duplicate_pairs(ph, dh, 10, 12)
    groups = union_find_groups(len(paths), pairs)
    assert len({groups[0], groups[6], groups[7], groups[8]}) == 1
    assert len(set(groups[1:6])) == 5 and groups[0] not in set(groups[1:6])
    # without dihedral hashing the flipped copy should not be matched
    ph2, dh2 = compute_hashes(paths, dihedral=False)
    g2 = union_find_groups(len(paths), find_duplicate_pairs(ph2, dh2, 10, 12))
    assert g2[6] != g2[0]


def test_group_split_keeps_groups_together_and_random_split_leaks():
    rng = np.random.default_rng(0)
    n_groups = 300
    groups = np.repeat(np.arange(n_groups), 4)
    y = np.array(["a", "b", "c"])[groups % 3]
    s = group_split(y, groups, 0.15, 0.15, 0)
    for g in range(n_groups):
        assert len(set(s[groups == g])) == 1
    frac_test = (s == "test").mean()
    assert 0.08 < frac_test < 0.22
    r = random_split(y, 0.15, 0.15, 0)
    shared = set(groups[r == "train"]) & set(groups[r == "test"])
    assert len(shared) > 0
    # leakage helper: fake hashes where each group has identical hashes
    ph = np.repeat(rng.integers(0, 2**63, n_groups, dtype=np.uint64), 4)[:, None].repeat(8, 1)
    assert cross_split_leakage(ph, ph, s, 0, 0) == 0
    assert cross_split_leakage(ph, ph, r, 0, 0) > 0.5


def test_build_catalog(tmp_path):
    for c in ("knife", "toy", "misc"):
        (tmp_path / "unsafe" / c).mkdir(parents=True)
        _img(1).save(tmp_path / "unsafe" / c / "a.jpg")
    df = build_catalog(tmp_path, LABELS)
    assert set(df.category) == {"knife", "toy", "unsafe"}  # misc falls back to the generic parent
    assert df.set_index("category").loc["toy", "label"] == "safe"
    assert isinstance(df, pd.DataFrame)


def test_corruptions_shape_and_severity_zero():
    img = np.asarray(_img(3, 64))
    for k in KINDS:
        out = corrupt(img, k, 3, seed=0)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert (corrupt(img, k, 0) == img).all()
