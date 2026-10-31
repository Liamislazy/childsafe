import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from childsafe.vision.detection import Detection
from childsafe.web.play_area import PlayAreaScanner, create_play_area_app


class Detector:
    names = ['person', 'knife', 'fork', 'scissors', 'spoon', 'cell phone']
    ok = True
    weights_file = 'fixture.pt'
    checksum = 'test-hash'
    device = 'fixture'

    def __init__(self, detections=()):
        self.detections = detections
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        self.shape = rgb.shape
        return self.detections


def image(flat=False):
    rgb = np.full((600, 800, 3), 245, dtype=np.uint8) if flat else np.random.default_rng(7).integers(30, 225, (600, 800, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format='PNG')
    return out.getvalue()


def test_visible_object_needs_no_face_or_hand_and_roi_coordinates_are_restored():
    detector = Detector([Detection((10, 20, 100, 200), 'scissors', .8),
                         Detection((150, 20, 200, 100), 'cell phone', .95)])
    result = PlayAreaScanner({}, detector).scan(image(), .1)
    assert detector.shape == (480, 640, 3)
    assert result['candidates'][0]['box'] == [90, 80, 180, 260]
    assert [d['label'] for d in result['candidates']] == ['scissors']
    assert result['usable_for_comparison']
    assert not result['calibrated'] and not result['images_saved']


def test_two_adjacent_same_class_objects_remain_distinct_and_duplicates_are_removed():
    detector = Detector([Detection((10, 20, 100, 200), 'scissors', .8),
                         Detection((12, 20, 100, 200), 'scissors', .7),
                         Detection((150, 20, 200, 200), 'scissors', .9)])
    assert len(PlayAreaScanner({}, detector).scan(image(), .1)['candidates']) == 2


@pytest.mark.parametrize('bad', [Detection((float('nan'), 1, 10, 10), 'knife', .8),
                                Detection((1, 1, 10, 10), 'knife', float('nan')),
                                Detection((1, 1, 10, 10), 'knife', .2),
                                Detection((20, 20, 1, 1), 'knife', .9)])
def test_invalid_or_low_confidence_candidates_do_not_become_checklist_items(bad):
    assert PlayAreaScanner({}, Detector([bad])).scan(image(), 0)['candidates'] == []


def test_failure_during_inference_never_supports_absence():
    class Fails(Detector):
        def detect(self, rgb):
            self.ok = False
            return []
    result = PlayAreaScanner({}, Fails()).scan(image(), 0)
    assert result['status'] == 'unavailable'
    assert not result['usable_for_comparison']


def test_flat_or_overexposed_image_cannot_clear_a_previous_candidate():
    result = PlayAreaScanner({}, Detector()).scan(image(flat=True), 0)
    assert not result['usable_for_comparison']
    assert result['quality']['reasons']


def test_detector_with_wrong_class_inventory_cannot_claim_supported_absence():
    detector = Detector();detector.names = ['person', 'spoon']
    result = PlayAreaScanner({}, detector).scan(image(), 0)
    assert not result['usable_for_comparison']


def test_empty_valid_view_is_not_a_safety_certificate():
    result = PlayAreaScanner({}, Detector()).scan(image(), 0)
    assert result['status'] == 'no_candidates'
    assert 'thủ công' in result['message']
    assert 'đã an toàn' not in result['message']


def test_api_origin_body_validation_and_real_page(tmp_path):
    with TestClient(create_play_area_app({}, Detector())) as client:
        assert client.get('/play-area').status_code == 200
        assert client.get('/play-area.js').status_code == 200
        assert client.get('/play-area').headers['cache-control'] == 'no-store'
        assert client.get('/play-area.js').headers['cache-control'] == 'no-store'
        assert client.get('/api/play-area/info').json()['supported'][0]['label'] == 'knife'
        assert client.post('/api/play-area/scan', content=image()).status_code == 403
        headers = {'x-childsafe-scan': '1'}
        assert client.post('/api/play-area/scan', content=b'bad', headers=headers).status_code == 422
        assert client.post('/api/play-area/scan?inset=.5', content=image(), headers=headers).status_code == 422
        assert client.post('/api/play-area/scan', content=image(), headers={**headers, 'origin': 'https://other.example'}).status_code == 403
        assert client.post('/api/play-area/scan', content=b'x'*(8*1024*1024+1), headers=headers).status_code == 413
        result = client.post('/api/play-area/scan?inset=.1', content=image(), headers=headers)
        assert result.status_code == 200
        assert 'image' not in result.json() and not result.json()['images_saved']
        assert not list(tmp_path.iterdir())


def test_health_probe_does_not_load_model(monkeypatch):
    def fail_load(self):
        raise AssertionError('Health probe must work before model initialization')
    monkeypatch.setattr(PlayAreaScanner, '_load', fail_load)
    with TestClient(create_play_area_app({})) as client:
        response = client.get('/api/play-area/health')
        assert response.json() == {'service': 'childsafe-play-area', 'status': 'running'}
        assert response.headers['cache-control'] == 'no-store'


def test_route_is_available_in_existing_runtime_app():
    from childsafe.web.app import create_app
    with TestClient(create_app(SimpleNamespace(cfg={}))) as client:
        assert client.get('/play-area').status_code == 200
        assert client.get('/play-area.js').status_code == 200
