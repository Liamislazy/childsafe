"""Download full Open Images photos with their human-drawn boxes for the hazard object detector.

Uses the box CSVs already in data/external/openimages/meta (scripts/fetch_openimages.py fetched
them). For each photo: a JPEG (longest side <= --max-side) and an adjacent YOLO detection .txt
with every non-group, non-drawing box of the target classes. Photos with a group-of or drawing
box of a target class are skipped (their individual boxes are incomplete), as are rotated photos.

Writes a manifest in the docs/hazard_annotations_template.csv format with reviewed and
annotation_complete set to false: Open Images may leave a target object unboxed when that class
was not verified for the photo, so a person reviews contact sheets before prepare_hazard_dataset.py.
Attribution (author, license, original URL) is kept per photo.

Run: python scripts/collect_openimages_detector.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data' / 'external' / 'openimages' / 'meta'
IMG_URL = 'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg'
# Open Images class -> configs/hazard_classes.yaml name
CLASSES = {'Knife': 'knife', 'Kitchen knife': 'knife', 'Fork': 'fork', 'Scissors': 'scissors', 'Pen': 'pen'}
SPLITS = {'train': 'train', 'validation': 'val', 'test': 'test'}
FILES = {'train': ('oidv6-train-annotations-bbox.csv', 'train-images-boxable-with-rotation.csv'),
         'validation': ('validation-annotations-bbox.csv', 'validation-images-with-rotation.csv'),
         'test': ('test-annotations-bbox.csv', 'test-images-with-rotation.csv')}
SOURCE = 'Open Images V6 boxes (human-drawn), via scripts/collect_openimages_detector.py'


def class_ids():
    names = yaml.safe_load((ROOT / 'configs' / 'hazard_classes.yaml').read_text(encoding='utf-8'))['names']
    return {name: int(i) for i, name in names.items()}


def select(split: str) -> pd.DataFrame:
    cls = pd.read_csv(META / 'class-descriptions-boxable.csv', header=None, names=['mid', 'name'])
    mids = dict(zip(cls.mid, cls.name))
    wanted = {m for m, n in mids.items() if n in CLASSES}
    boxes_file, images_file = FILES[split]
    parts = []
    for chunk in pd.read_csv(META / boxes_file, chunksize=2_000_000,   # the train file is 2.3 GB
                             usecols=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf', 'IsDepiction']):
        parts.append(chunk[chunk.LabelName.isin(wanted)])
    b = pd.concat(parts, ignore_index=True)
    incomplete = set(b.ImageID[(b.IsGroupOf == 1) | (b.IsDepiction == 1)])
    b = b[~b.ImageID.isin(incomplete)].copy()
    b['name'] = b.LabelName.map(mids).map(CLASSES)
    imgs = pd.read_csv(META / images_file, usecols=['ImageID', 'OriginalURL', 'License', 'Author', 'Rotation'])
    imgs = imgs[imgs.Rotation.fillna(0) == 0]   # skip rotated photos rather than guess the box frame
    return b.merge(imgs, on='ImageID')


def download(job):
    split, image_id, lines, out, max_side = job
    image, label = out / f'{image_id}.jpg', out / f'{image_id}.txt'
    if image.exists() and label.exists():
        return 'exists'
    err = None
    for attempt in range(4):   # network drops happen on long runs: retry with backoff
        try:
            with urllib.request.urlopen(IMG_URL.format(split=split, image_id=image_id), timeout=30) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert('RGB')
            break
        except Exception as e:
            err = e
            time.sleep(2 * 2 ** attempt)
    else:
        return f'download failed: {type(err).__name__}'
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)   # boxes are relative, so unchanged
    img.save(image, quality=92)
    label.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'detector' / 'openimages')
    ap.add_argument('--splits', nargs='*', default=list(SPLITS))
    ap.add_argument('--max-side', type=int, default=1280)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--dry-run', action='store_true', help='only count photos and boxes')
    args = ap.parse_args()
    ids = class_ids()
    jobs, rows = [], []
    for split in args.splits:
        b = select(split)
        per_class = b.groupby('name').ImageID.nunique().to_dict()
        print(f'{split}: {b.ImageID.nunique()} photos, {len(b)} boxes; photos per class {per_class}')
        out = args.out / SPLITS[split]
        for image_id, g in b.groupby('ImageID'):
            lines = [f"{ids[r.name]} {(r.XMin + r.XMax) / 2:.6f} {(r.YMin + r.YMax) / 2:.6f} "
                     f"{r.XMax - r.XMin:.6f} {r.YMax - r.YMin:.6f}" for r in g.itertuples()]
            first = g.iloc[0]
            jobs.append((split, image_id, lines, out, args.max_side))
            rows.append(dict(image=(Path('openimages') / SPLITS[split] / f'{image_id}.jpg').as_posix(),
                             split=SPLITS[split], group_id=f'oi:{image_id}', reviewed='false',
                             annotation_complete='false', source=SOURCE, license=first.License,
                             author=first.Author, url=first.OriginalURL))
    if args.dry_run:
        return
    for split in args.splits:
        (args.out / SPLITS[split]).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        status = list(pool.map(download, jobs))
    kept = [row for row, s in zip(rows, status) if s in ('ok', 'exists')]
    manifest = args.out.parent / 'openimages_manifest.csv'
    with manifest.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)
    failed = len(status) - len(kept)
    print(f'{len(kept)} photos saved, {failed} failed; manifest {manifest}')


if __name__ == '__main__':
    main()
