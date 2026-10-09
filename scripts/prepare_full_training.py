"""Resolve the pinned full dataset and reserve validation groups for calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_FOLDERS = {'childsun': None, 'openimages': 'validation', 'openimages_train': 'train'}


def prepare_manifest(base, raw_root, openimages_root, seed=42):
    df = base.copy()
    required = {'path', 'source', 'category', 'label', 'group', 'source_id', 'split_group'}
    if not required.issubset(df.columns) or set(df.source) != set(SOURCE_FOLDERS):
        raise ValueError('Use splits_with_other_train.csv with all three sources')
    if not set(df.split_group).issubset({'train', 'val', 'test'}):
        raise ValueError('Expected the original train/val/test group split')
    if df.groupby('group').split_group.nunique().max() != 1:
        raise ValueError('Group leakage in pinned split')
    if df.groupby(['source', 'source_id']).split_group.nunique().max() != 1:
        raise ValueError('Source-photo leakage in pinned split')
    if df.duplicated(['source', 'path']).any():
        raise ValueError('Duplicate source/path rows')
    roots = {'childsun': Path(raw_root).resolve(),
             **{s: Path(openimages_root).resolve()/folder
                for s, folder in SOURCE_FOLDERS.items() if folder}}
    for source, part in df.groupby('source'):
        root = roots[source]
        expected = set(part.path)
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}}
        if expected != actual:
            raise ValueError(f'{source}: pinned split differs from dataset: '
                             f'{len(expected-actual)} missing, {len(actual-expected)} extra. '
                             'Check input root and upload the complete matching dataset.')
    download = pd.read_csv(Path(openimages_root)/'manifest.csv')
    if not {'train', 'validation', 'test'}.issubset(set(download['split'])):
        raise ValueError('Open Images input must include train, validation, test and manifest.csv')
    successful = download[download.status.isin(['ok', 'exists'])]
    for source, folder in SOURCE_FOLDERS.items():
        if folder:
            reported = set(successful.loc[successful['split'] == folder, 'path'])
            expected = {f'{folder}/{p}' for p in df.loc[df.source == source, 'path']}
            if reported != expected:
                raise ValueError(f'{folder}: download manifest differs from pinned split')
    test_rows = successful[successful['split'] == 'test']
    if test_rows.empty or any(not (Path(openimages_root)/p).is_file() for p in test_rows.path):
        raise ValueError('Open Images test crops are missing')
    training_ids = set(successful.loc[successful['split'].isin(['train', 'validation']), 'image_id'])
    if training_ids & set(test_rows.image_id):
        raise ValueError('Open Images source-photo overlap with external test')
    val = df[df.split_group == 'val']
    classes = set(df.category)
    for category in classes:
        if val.loc[val.category == category, 'group'].nunique() < 2:
            raise ValueError(f'{category}: need at least two validation groups')
    rng = np.random.default_rng(seed)
    groups = sorted(val.group.unique())
    for _ in range(1000):
        cal_groups = set(rng.permutation(groups)[:len(groups)//2])
        in_cal = val.group.isin(cal_groups)
        if set(val.loc[in_cal, 'category']) == classes == set(val.loc[~in_cal, 'category']):
            df.loc[(df.split_group == 'val') & df.group.isin(cal_groups), 'split_group'] = 'cal'
            break
    else:
        raise ValueError('Cannot reserve calibration groups with all classes')
    for role in ('train', 'val', 'cal', 'test'):
        if set(df.loc[df.split_group == role, 'category']) != classes:
            raise ValueError(f'{role} does not cover every class')
    df['original_path'] = df.path
    df['path'] = [str(roots[s]/p) for s, p in zip(df.source, df.path)]
    df['domain'] = df.source
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='data/splits/splits_with_other_train.csv')
    ap.add_argument('--raw-root', required=True)
    ap.add_argument('--openimages-root', required=True)
    ap.add_argument('--out', default='data/splits/full_training.csv')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)
    base = Path(args.base)
    df = prepare_manifest(pd.read_csv(base), args.raw_root, args.openimages_root, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    report = {'seed': args.seed, 'base_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
              'images': len(df), 'classes': sorted(df.category.unique()),
              'counts': {' / '.join(k): int(v) for k, v in
                         df.groupby(['source', 'split_group']).size().items()},
              'external_test': 'Open Images test is excluded from the training manifest'}
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(df.groupby(['source', 'split_group']).size().to_string())
    print(f'Wrote {out}: {len(df)} images, {df.category.nunique()} classes')


if __name__ == '__main__':
    main()
