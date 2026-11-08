import base64
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from childsafe.web.label_tool import create_label_app

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_detector_manifest import build


def jpeg():
    out = io.BytesIO()
    Image.fromarray(np.full((48, 64, 3), 120, np.uint8)).save(out, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()


def capture(**kw):
    body = dict(image=jpeg(), person='P1', session='S1', consent=True, complete=True,
                boxes=[dict(label='knife', x0=.1, y0=.2, x1=.5, y1=.6)])
    body.update(kw)
    return body


def test_capture_saves_photo_label_and_manifest(tmp_path):
    client = TestClient(create_label_app(tmp_path))
    saved = client.post('/api/capture', json=capture()).json()
    image = tmp_path / saved['saved']
    assert image.is_file() and image.with_suffix('.txt').read_text().split() == ['0', '0.300000', '0.400000', '0.400000', '0.400000']
    rows = list(csv.DictReader((tmp_path / 'own_manifest.csv').open(encoding='utf-8')))
    assert rows[0]['group_id'] == 'person:P1' and rows[0]['reviewed'] == 'false' and rows[0]['split'] == ''
    background = client.post('/api/capture', json=capture(boxes=[])).json()
    assert (tmp_path / background['saved']).with_suffix('.txt').read_text() == ''
    assert client.get('/api/info').json()['own_background'] == 1
    assert client.delete('/api/capture', params={'image': background['saved']}).status_code == 200
    assert not (tmp_path / background['saved']).exists()


def test_capture_refuses_missing_consent_bad_boxes_and_paths(tmp_path):
    client = TestClient(create_label_app(tmp_path))
    assert client.post('/api/capture', json=capture(consent=False)).status_code == 400
    assert client.post('/api/capture', json=capture(complete=False)).status_code == 400
    assert client.post('/api/capture', json=capture(person='../x')).status_code == 400
    assert client.post('/api/capture', json=capture(boxes=[dict(label='spoon', x0=.1, y0=.1, x1=.2, y1=.2)])).status_code == 400
    assert client.post('/api/capture', json=capture(boxes=[dict(label='fork', x0=.5, y0=.1, x1=.2, y1=.2)])).status_code == 400
    assert client.post('/api/capture', json=capture(image='data:image/jpeg;base64,bm90IGFuIGltYWdl')).status_code == 422
    (tmp_path / 'secret.txt').write_text('x')
    assert client.get('/files/../secret.txt').status_code == 404
    assert client.get('/files/secret.txt').status_code == 404
    assert not (tmp_path / 'own').exists()


def test_review_then_build_keeps_ok_rows_and_splits_by_person(tmp_path):
    client = TestClient(create_label_app(tmp_path))
    a = client.post('/api/capture', json=capture(person='P1')).json()['saved']
    b = client.post('/api/capture', json=capture(person='P2')).json()['saved']
    c = client.post('/api/capture', json=capture(person='P2')).json()['saved']
    page = client.get('/api/review', params={'set': 'own'}).json()
    assert page['total'] == 3 and page['items'][0]['boxes'][0]['label'] == 'knife'
    client.post('/api/review', json={'set': 'own', 'verdicts': {a: 'ok', b: 'ok', c: 'reject'}})
    assert client.get('/api/review', params={'set': 'own'}).json()['total'] == 0
    assert client.get('/files/' + a).status_code == 200
    report = build(tmp_path, {'P1': 'val', 'P2': 'test'})
    assert report['rows'] == {'train': 0, 'val': 1, 'test': 1} and report['skipped']['not ok'] == 1
    rows = list(csv.DictReader((tmp_path / 'annotations.csv').open(encoding='utf-8')))
    assert {r['reviewed'] for r in rows} == {'true'} and 'verdict' not in rows[0]
