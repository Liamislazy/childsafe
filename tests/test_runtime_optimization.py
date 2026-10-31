"""Real regressions: changing held objects, repeated ROI work and overlapping web requests."""
import asyncio
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection, SegmentationDetector
from childsafe.web.app import create_app
from childsafe.runtime import configure_runtime


class Face:
    def detect(self, rgb, t):
        face = np.zeros((478, 3))
        face[234, :2] = [20, 40]; face[454, :2] = [100, 40]
        face[[78, 13, 308, 14], :2] = [[50, 50], [60, 45], [70, 50], [60, 55]]
        face[[33, 133], :2] = [40, 30]; face[[362, 263], :2] = [80, 30]
        hand = np.column_stack((np.linspace(50, 70, 21), np.linspace(50, 65, 21), np.zeros(21)))
        return SimpleNamespace(t=t, width=rgb.shape[1], height=rgb.shape[0], face=face,
                               hands=[hand], face_matrix=np.eye(4))

    def close(self): pass


class Detector:
    ok, supports_roi = True, True
    label = 'knife'
    confidence = .99
    width = 15
    def __init__(self): self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        mask = np.zeros(rgb.shape[:2], bool); mask[45:60, 65:65+self.width] = True
        return [Detection((0, 0, 160, 160), 'person', .99),
                Detection((65, 45, 65+self.width, 60), self.label, self.confidence, mask)]


def result(label='knife'):
    probability = .99 if label == 'knife' else .01
    return dict(p_unsafe_raw=probability, p_unsafe_cal=probability, obj_class=label,
                obj_conf=.99, p_unknown_cal=0., top3=[])


def configuration():
    cfg = load_config()
    cfg['quality']['min_sharpness'] = 0
    cfg['smoothing']['method'] = 'none'
    cfg['detector']['roi_min_px'] = 64
    return cfg


RGB = np.full((160, 160, 3), 140, np.uint8)


def test_spoon_replacing_knife_at_same_position_invalidates_classification_and_dwell():
    detector = Detector(); calls = []
    def classify(*args):
        calls.append(detector.label)
        return result(detector.label)
    pipe = FramePipeline(configuration(), None, detector=detector, landmark_factory=Face, classifier=classify)
    try:
        for i in range(20): out = pipe.process(RGB, i*.05)[2]
        assert out['level'] == 2
        before = len(calls)
        detector.label = 'spoon'
        out = pipe.process(RGB, .97)[2]
        assert len(calls) == before+1
        assert out['persons'][0]['obj_class'] == 'spoon'
        assert out['persons'][0]['level'] != 2
        assert out['feature_rows'][0]['history_reset']
    finally: pipe.close()


def test_resolved_mask_skips_extra_detector_but_weak_mask_still_requests_zoom():
    detector = Detector()
    pipe = FramePipeline(configuration(), None, detector=detector, landmark_factory=Face, classifier=lambda *a: result())
    try:
        scene, row, _ = pipe.process(RGB, 0.)
        assert detector.calls == 1 and row['mask_reliable']
        detector.confidence = .4
        pipe.process(RGB, .1)
        assert detector.calls == 3  # full frame + the missing confidence's zoom pass
    finally: pipe.close()


def test_classifier_failure_clears_old_unsafe_cache_and_recovers():
    fail = [False]
    def classify(*args):
        if fail[0]: raise RuntimeError('private data')
        return result()
    cfg = configuration(); cfg['cascade']['classify_interval_s'] = .1
    pipe = FramePipeline(cfg, None, detector=Detector(), landmark_factory=Face, classifier=classify)
    try:
        pipe.process(RGB, 0.)
        fail[0] = True
        _, row, out = pipe.process(RGB, .2)
        state = next(iter(pipe.states.values()))
        assert not row['classifier_ok'] and state['processor'].obj is None
        assert 'obj_result' not in state and 'classified' not in state
        fail[0] = False
        _, row, _ = pipe.process(RGB, .3)
        assert row['classifier_ok'] and not pipe.classifier_error
    finally: pipe.close()


def test_one_person_classifier_error_does_not_poison_another_person():
    class TwoPeople:
        ok = True
        def detect(self, rgb):
            out = []
            for offset in (0, 160):
                mask = np.zeros(rgb.shape[:2], bool); mask[45:60, offset+65:offset+80] = True
                out.extend([Detection((offset, 0, offset+160, 160), 'person', .99),
                            Detection((offset+65, 45, offset+80, 60), 'knife', .99, mask)])
            return out
    def classify(rgb, box):
        if box[0] < 160: raise RuntimeError('bad crop')
        return result()
    pipe = FramePipeline(configuration(), None, detector=TwoPeople(), landmark_factory=Face, classifier=classify)
    try:
        _, _, out = pipe.process(np.full((160, 320, 3), 140, np.uint8), 0.)
        rows = sorted(out['feature_rows'], key=lambda row: row['person_x0'])
        assert len(rows) == 2
        assert not rows[0]['classifier_ok'] and rows[1]['classifier_ok']
    finally: pipe.close()


def test_too_small_object_cannot_reuse_old_unsafe_prediction():
    detector = Detector()
    pipe = FramePipeline(configuration(), None, detector=detector, landmark_factory=Face, classifier=lambda *a: result())
    try:
        pipe.process(RGB, 0.)
        detector.width = 5
        _, row, out = pipe.process(RGB, .2)
        assert out['persons'][0]['obj_status'] == 'none'
        assert 'object crop too small for classification' in row['quality_reasons']
        assert next(iter(pipe.states.values()))['processor'].obj is None
    finally: pipe.close()


def test_inference_does_not_block_info_and_busy_frames_are_not_queued():
    entered, release = threading.Event(), threading.Event()
    class Demo:
        paths = {}
        cfg = load_config()
        def frame(self, *args):
            entered.set()
            assert release.wait(3)
            return {'ok': True}
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(Demo())),
                                     base_url='http://test') as client:
            pending = asyncio.create_task(client.post('/api/frame', content=b'frame'))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                info = await asyncio.wait_for(client.get('/api/info'), 1)
                assert info.status_code == 200
                busy = await asyncio.wait_for(client.post('/api/frame', content=b'new'), 1)
                assert busy.status_code == 503
            finally:
                release.set()
                assert (await pending).status_code == 200
    asyncio.run(scenario())


def test_cpu_thread_budget_is_cli_only_and_zero_leaves_default(monkeypatch):
    import torch
    calls = []
    monkeypatch.setattr(torch, 'set_num_threads', calls.append)
    configure_runtime({'runtime': {'cpu_threads': 2}})
    configure_runtime({'runtime': {'cpu_threads': 0}})
    assert calls == [2]
    for value in (-1, True, 2.5, '2'):
        with pytest.raises(ValueError): configure_runtime({'runtime': {'cpu_threads': value}})


def test_detector_restores_cpu_budget_after_lazy_backend_initialization(monkeypatch):
    import torch
    current = [2]
    monkeypatch.setattr(torch, 'get_num_threads', lambda: current[0])
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: current.__setitem__(0, n))
    class Model:
        def predict(self, *args, **kwargs):
            current[0] = 8
            return [SimpleNamespace(boxes=[], masks=None)]
    adapter = SegmentationDetector.__new__(SegmentationDetector)
    adapter.model, adapter.confidence, adapter.device = Model(), .5, 'cpu'
    adapter.cpu_threads = 2
    adapter.error = ''
    assert adapter.detect(RGB) == [] and adapter.ok
    assert current[0] == 2
