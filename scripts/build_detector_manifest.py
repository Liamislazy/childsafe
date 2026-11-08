"""Combine reviewed Open Images and own photos into data/detector/annotations.csv.

Keeps rows whose review verdict is 'ok'. Open Images photos keep their own split. Own photos get
their split from --split PERSON=train|val|test, so one person never spans two splits.
Then: python scripts/prepare_hazard_dataset.py --manifest data/detector/annotations.csv
      --out data/detector/hazards_v1 --task detect
"""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from childsafe.web.label_tool import FIELDS, MANIFESTS, read_rows


def build(data_root: Path, person_splits: dict[str, str]) -> dict:
    rows, skipped = [], {'not ok': 0, 'no split': 0}
    for name in MANIFESTS:
        for row in read_rows(data_root / MANIFESTS[name]):
            if row['verdict'] != 'ok':
                skipped['not ok'] += 1
                continue
            if name == 'own':
                row['split'] = person_splits.get(row['author'], '')
                if not row['split']:
                    skipped['no split'] += 1
                    continue
            rows.append(dict(row, reviewed='true', annotation_complete='true'))
    out = data_root / 'annotations.csv'
    with out.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[f for f in FIELDS if f != 'verdict'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    per_split = {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}
    return dict(manifest=str(out), rows=per_split, skipped=skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'detector')
    ap.add_argument('--split', action='append', default=[], metavar='PERSON=SPLIT',
                    help='split for one person code from the label tool, e.g. P1=val P2=test')
    args = ap.parse_args()
    person_splits = {}
    for item in args.split:
        person, _, split = item.partition('=')
        if split not in ('train', 'val', 'test'):
            ap.error(f'bad --split {item!r}; use PERSON=train|val|test')
        person_splits[person] = split
    print(build(args.data, person_splits))


if __name__ == '__main__':
    main()
