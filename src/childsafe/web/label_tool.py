"""Local photo capture, box labelling and review for the hazard object detector.

Consented adults only. One still photo per click, never video. Files stay under data/detector
(git-ignored) and nothing is uploaded. Manifests follow docs/hazard_annotations_template.csv plus
a 'verdict' column (ok/reject) set during review; scripts/build_detector_manifest.py keeps the
'ok' rows for prepare_hazard_dataset.py.
"""
from __future__ import annotations

import base64
import binascii
import csv
import datetime
import io
import re
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / 'static'
NO_CACHE = {'Cache-Control': 'no-store'}
FIELDS = ['image', 'split', 'group_id', 'reviewed', 'annotation_complete', 'source', 'license',
          'author', 'url', 'verdict']
MANIFESTS = {'own': 'own_manifest.csv', 'openimages': 'openimages_manifest.csv'}
LABELLED = ('knife', 'fork', 'scissors', 'pen')   # what the tool lets you draw
CODE = re.compile(r'[A-Za-z0-9_-]{1,20}')


def class_ids(classes=ROOT / 'configs' / 'hazard_classes.yaml'):
    names = yaml.safe_load(Path(classes).read_text(encoding='utf-8'))['names']
    return {name: int(i) for i, name in names.items()}


class Box(BaseModel):
    label: str
    x0: float
    y0: float
    x1: float
    y1: float


class Capture(BaseModel):
    image: str                 # data:image/jpeg;base64,...
    person: str
    session: str
    boxes: list[Box]
    consent: bool
    complete: bool


class Review(BaseModel):
    set: str
    verdicts: dict[str, str]   # image path -> ok | reject


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [dict({k: '' for k in FIELDS}, **row) for row in csv.DictReader(stream)]


def write_rows(path: Path, rows: list[dict]):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_boxes(label: Path, names: dict[int, str]) -> list[dict]:
    """YOLO detection lines -> corner boxes for display."""
    out = []
    for line in label.read_text(encoding='utf-8').splitlines() if label.is_file() else []:
        parts = line.split()
        if len(parts) == 5:
            c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
            out.append(dict(label=names.get(c, str(c)), x0=x - w / 2, y0=y - h / 2, x1=x + w / 2, y1=y + h / 2))
    return out


def create_label_app(data_root: Path = ROOT / 'data' / 'detector', classes=ROOT / 'configs' / 'hazard_classes.yaml'):
    data_root = Path(data_root)
    ids = class_ids(classes)
    names = {i: n for n, i in ids.items()}
    lock = threading.Lock()
    app = FastAPI(title='ChildSafe — gán nhãn detector')
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    def manifest(name: str) -> Path:
        if name not in MANIFESTS:
            raise ValueError('unknown set')
        return data_root / MANIFESTS[name]

    @app.get('/')
    def page():
        return FileResponse(STATIC / 'label_tool.html', headers=NO_CACHE)

    @app.get('/label-tool.js')
    def javascript():
        return FileResponse(STATIC / 'label_tool.js', media_type='text/javascript', headers=NO_CACHE)

    @app.get('/api/info')
    def info():
        counts = {}
        for name in MANIFESTS:
            rows = read_rows(manifest(name))
            counts[name] = dict(total=len(rows), reviewed=sum(r['verdict'] in ('ok', 'reject') for r in rows),
                                rejected=sum(r['verdict'] == 'reject' for r in rows))
        own = read_rows(manifest('own'))
        per_class = {c: 0 for c in LABELLED}
        background = 0
        for row in own:
            boxes = read_boxes((data_root / row['image']).with_suffix('.txt'), names)
            background += not boxes
            for b in boxes:
                per_class[b['label']] = per_class.get(b['label'], 0) + 1
        return JSONResponse(dict(classes=list(LABELLED), counts=counts, own_boxes=per_class,
                                 own_background=background), headers=NO_CACHE)

    @app.post('/api/capture')
    def capture(body: Capture):
        if not body.consent:
            return JSONResponse({'error': 'Cần xác nhận người trong ảnh là người lớn đã đồng ý.'}, 400)
        if not body.complete:
            return JSONResponse({'error': 'Cần xác nhận đã gán hết vật trong ảnh (để trống nếu không có).'}, 400)
        if not (CODE.fullmatch(body.person) and CODE.fullmatch(body.session)):
            return JSONResponse({'error': 'Mã người và mã buổi chỉ gồm chữ, số, - và _.'}, 400)
        lines = []
        for b in body.boxes:
            if b.label not in LABELLED or not (0 <= b.x0 < b.x1 <= 1 and 0 <= b.y0 < b.y1 <= 1):
                return JSONResponse({'error': 'Khung hoặc loại vật không hợp lệ.'}, 400)
            lines.append(f'{ids[b.label]} {(b.x0 + b.x1) / 2:.6f} {(b.y0 + b.y1) / 2:.6f} '
                         f'{b.x1 - b.x0:.6f} {b.y1 - b.y0:.6f}')
        try:
            raw = base64.b64decode(body.image.split(',', 1)[-1], validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format != 'JPEG':
                    raise ValueError
                im.load()
        except (binascii.Error, ValueError, UnidentifiedImageError, OSError):
            return JSONResponse({'error': 'Ảnh không hợp lệ.'}, 422)
        now = datetime.datetime.now()
        rel = Path('own') / body.person / body.session / f'{now:%Y%m%d_%H%M%S_%f}.jpg'
        with lock:
            target = data_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            target.with_suffix('.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            rows = read_rows(manifest('own'))
            # Same person, any session = one group, so a person never spans train/val/test.
            rows.append(dict(image=rel.as_posix(), split='', group_id=f'person:{body.person}', reviewed='false',
                             annotation_complete='true', source=f'own webcam photo, session {body.session}',
                             license=f'consenting adult, confirmed in label tool {now:%Y-%m-%d}',
                             author=body.person, url='', verdict=''))
            write_rows(manifest('own'), rows)
        return {'saved': rel.as_posix(), 'boxes': len(lines)}

    @app.delete('/api/capture')
    def remove(image: str):
        with lock:
            rows = read_rows(manifest('own'))
            keep = [r for r in rows if r['image'] != image]
            if len(keep) == len(rows) or not image.startswith('own/'):
                return JSONResponse({'error': 'Không tìm thấy ảnh.'}, 404)
            for path in ((data_root / image), (data_root / image).with_suffix('.txt')):
                path.unlink(missing_ok=True)
            write_rows(manifest('own'), keep)
        return {'removed': image}

    @app.get('/api/review')
    def review_page(set: str, offset: int = 0, limit: int = 24, todo: bool = True):
        rows = [r for r in read_rows(manifest(set)) if not todo or r['verdict'] not in ('ok', 'reject')]
        items = [dict(image=r['image'], verdict=r['verdict'],
                      boxes=read_boxes((data_root / r['image']).with_suffix('.txt'), names))
                 for r in rows[offset:offset + min(max(limit, 1), 60)]]
        return JSONResponse(dict(total=len(rows), items=items), headers=NO_CACHE)

    @app.post('/api/review')
    def review(body: Review):
        if any(v not in ('ok', 'reject') for v in body.verdicts.values()):
            return JSONResponse({'error': 'verdict must be ok or reject'}, 400)
        with lock:
            rows = read_rows(manifest(body.set))
            for row in rows:
                verdict = body.verdicts.get(row['image'])
                if verdict:
                    row['verdict'] = verdict
                    row['reviewed'] = 'true'
                    if verdict == 'ok':
                        row['annotation_complete'] = 'true'
            write_rows(manifest(body.set), rows)
        return {'updated': len(body.verdicts)}

    @app.get('/files/{rel:path}')
    def files(rel: str):
        path = (data_root / rel).resolve()
        if not path.is_relative_to(data_root.resolve()) or path.suffix.lower() != '.jpg' or not path.is_file():
            return JSONResponse({'error': 'not found'}, 404)
        return FileResponse(path)

    return app
