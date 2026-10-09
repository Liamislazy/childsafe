"""Download attributed, object-only Open Images validation crops for manual review.

No scraped search results, no credentials, no child/face images. Never auto-approve labels.
Run: python scripts/collect_openimages.py --per-class 60 --out data/supplement
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw

BASE = 'https://storage.googleapis.com/openimages/'
LABELS = {'pen': '/m/0k1tl', 'coin': '/m/0242l', 'knife': '/m/04ctx',
          'scissors': '/m/01lsmm', 'fork': '/m/0dt3t', 'spoon': '/m/0cmx8'}
RISKS = {c: ('safe' if c == 'spoon' else 'unsafe') for c in LABELS}

def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/supplement')
    ap.add_argument('--per-class', type=int, default=60)
    ap.add_argument('--subset', choices=['validation','test'], default='validation')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    def metadata(name, url):
        path = out / name
        if not path.exists():
            path.write_bytes(fetch(url))
        return csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig')))
    descriptions = dict(csv.reader(io.StringIO(fetch(BASE + 'v5/class-descriptions-boxable.csv').decode())))
    excluded = {k for k, v in descriptions.items() if v in
                {'Person', 'Man', 'Woman', 'Boy', 'Girl', 'Human face', 'Human head'}}
    subset=args.subset
    boxes = list(metadata(f'{subset}-boxes.csv', BASE + f'v5/{subset}-annotations-bbox.csv'))
    people = {b['ImageID'] for b in boxes if b['LabelName'] in excluded}
    meta = {r['ImageID']: r for r in metadata(f'{subset}-images.csv',
             BASE + f'2018_04/{subset}/{subset}-images-with-rotation.csv')}
    selected = []
    counts = {}
    for category, label in LABELS.items():
        used = set()
        candidates = sorted((b for b in boxes if b['LabelName'] == label), key=lambda b: b['ImageID'])
        for b in candidates:
            image_id = b['ImageID']; m = meta.get(image_id, {})
            if image_id in people or image_id in used or 'creativecommons.org/licenses/by/' not in m.get('License', ''):
                continue
            if any(b.get(flag, '0') != '0' for flag in ('IsGroupOf', 'IsDepiction', 'IsInside')):
                continue
            used.add(image_id); selected.append((category, b, m))
            if len(used) >= args.per_class:
                break
        counts[category] = len(used)
    print('Candidates (before download):', counts, flush=True)
    def download(item):
        category, b, m = item; image_id = b['ImageID']
        url = f'https://open-images-dataset.s3.amazonaws.com/{subset}/{image_id}.jpg'
        relative = f'images/{category}/{image_id}.jpg'; dest = out / relative
        try:
            if not dest.exists():
                image = Image.open(io.BytesIO(fetch(url))).convert('RGB')
                w, h = image.size
                x0, x1 = float(b['XMin']) * w, float(b['XMax']) * w
                y0, y1 = float(b['YMin']) * h, float(b['YMax']) * h
                pad = .08 * max(x1-x0, y1-y0)
                crop = image.crop((max(0, int(x0-pad)), max(0, int(y0-pad)),
                                   min(w, int(x1+pad)), min(h, int(y1+pad))))
                if min(crop.size) < 20:
                    return None
                dest.parent.mkdir(parents=True, exist_ok=True); crop.save(dest, quality=95)
            return {'path': relative, 'category': category, 'label': RISKS[category],
                    'group': f'openimages:{subset}:{image_id}', 'domain': 'openimages_object_crop',
                    'reviewed': 'false', 'image_id': image_id, 'download_url': url,
                    'original_url': m.get('OriginalURL', ''), 'landing_url': m.get('OriginalLandingURL', ''),
                    'license': m.get('License', ''), 'author': m.get('Author', ''),
                    'title': m.get('Title', ''), 'bbox': json.dumps([b[k] for k in ('XMin','YMin','XMax','YMax')])}
        except Exception as exc:
            print(f'Skipped {category}/{image_id}: {exc}', flush=True)
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for r in pool.map(download, selected) if r]
    if not rows:
        raise RuntimeError('No images downloaded; do not train an empty supplement.')
    with (out/'manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    sheets = out/'review'; sheets.mkdir(exist_ok=True)
    for category in LABELS:
        subset = [r for r in rows if r['category'] == category]
        for page in range(0, len(subset), 40):
            part = subset[page:page+40]
            sheet = Image.new('RGB', (1000, 150*((len(part)+4)//5)), 'white'); draw=ImageDraw.Draw(sheet)
            for i, r in enumerate(part):
                tile = ImageOps.contain(Image.open(out/r['path']).convert('RGB'), (190,120))
                x,y=(i%5)*200,(i//5)*150; sheet.paste(tile,(x,y))
                draw.text((x,y+121), r['image_id'], fill='black')
            sheet.save(sheets/f'{category}_{page//40+1}.jpg')
    summary = {'downloaded': {c: sum(r['category']==c for r in rows) for c in LABELS},
               'reviewed': 0, 'source': f'Open Images {subset}',
               'warning': 'Pending manual label/privacy review. No person annotations is not proof of no person.'}
    (out/'collection_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
