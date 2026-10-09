"""Regression tests for live-camera speed fixes (2026-10-09)."""
import math
from types import SimpleNamespace

import numpy as np

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection, SegmentationDetector
from childsafe.vision.elevated import ElevatedQuality, mask_region_distance


def _old_mask_region_distance(mask, polygon):
    """The previous all-pixels implementation, kept here as the reference."""
    polygon = np.asarray(polygon, float)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return math.nan
    a, b = polygon, np.roll(polygon, -1, axis=0)
    seg = b - a
    pts = np.column_stack((xs, ys))
    delta = pts[:, None, :] - a[None]
    cross = seg[None, :, 0] * delta[:, :, 1] - seg[None, :, 1] * delta[:, :, 0]
    if (np.all(cross >= -1e-8, 1) | np.all(cross <= 1e-8, 1)).any():
        return 0.
    proj = np.clip((delta * seg[None]).sum(2) / np.maximum((seg * seg).sum(1), 1e-9), 0, 1)
    return float(np.linalg.norm(delta - proj[:, :, None] * seg[None], axis=2).min())


def test_mask_distance_matches_all_pixel_reference():
    rng = np.random.default_rng(0)
    for _ in range(60):
        mask = np.zeros((120, 160), bool)
        for _ in range(rng.integers(1, 4)):
            x0, y0 = rng.integers(0, 140), rng.integers(0, 100)
            mask[y0:y0 + rng.integers(2, 30), x0:x0 + rng.integers(2, 30)] = True
        cx, cy = rng.uniform(-20, 180), rng.uniform(-20, 140)
        poly = [[cx - 8, cy], [cx, cy - 3], [cx + 8, cy], [cx, cy + 3]]
        new, old = mask_region_distance(mask, poly), _old_mask_region_distance(mask, poly)
        assert (new == 0) == (old == 0)
        assert abs(new - old) < 1.0
    assert math.isnan(mask_region_distance(np.zeros((10, 10), bool), [[1, 1], [3, 1], [3, 3]]))


def test_quality_measure_matches_previous_formula():
    rgb = np.random.default_rng(1).integers(0, 256, (90, 130, 3), dtype=np.uint8)
    out = ElevatedQuality(load_config()['quality']).measure(rgb, None, {})
    gray = rgb.astype(np.float32).mean(2)
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    assert abs(out['brightness'] - gray.mean()) < 1e-3
    assert abs(out['sharpness'] - lap.var()) / lap.var() < 1e-4


class _Landmarks:
    def detect(self, rgb, t):
        return SimpleNamespace(t=t, width=rgb.shape[1], height=rgb.shape[0], face=None, hands=[], face_matrix=None)

    def close(self):
        pass


def test_overlapping_people_do_not_create_a_landmark_model_per_frame():
    made = []

    def factory():
        made.append(1)
        return _Landmarks()

    class Overlap:
        ok = True
        def detect(self, rgb):   # IoU > 0.1 -> tracker marks both ambiguous -> new IDs every frame
            return [Detection((10, 10, 120, 200), 'person', .9), Detection((60, 10, 170, 200), 'person', .9)]

    pipe = FramePipeline(load_config(), None, detector=Overlap(), landmark_factory=factory,
                         classifier=lambda *a: None)
    rgb = np.zeros((220, 200, 3), np.uint8)
    try:
        ids = set()
        for i in range(20):
            _, _, out = pipe.process(rgb, i * .1)
            ids.update(p['person_id'] for p in out['persons'])
        assert len(ids) > 10                       # identities are still discarded (safety rule kept)
        assert len(made) <= 4                      # ...but MediaPipe instances are reused, not recreated
    finally:
        pipe.close()


def test_classifier_reused_for_same_object_within_interval():
    calls = []

    def classifier(rgb, box):
        calls.append(box)
        return {'p_unsafe_raw': .9, 'p_unsafe_cal': .9, 'obj_class': 'knife', 'obj_conf': .9,
                'p_unknown_cal': 0., 'top3': []}

    class Hands(_Landmarks):
        def detect(self, rgb, t):
            out = super().detect(rgb, t)
            out.hands = [np.column_stack((np.linspace(40, 60, 21), np.linspace(40, 60, 21), np.zeros(21)))]
            return out

    class One:
        ok = True
        def detect(self, rgb):
            mask = np.zeros(rgb.shape[:2], bool); mask[40:80, 50:70] = True
            return [Detection((0, 0, 200, 200), 'person', .9), Detection((50, 40, 70, 80), 'knife', .9, mask)]

    cfg = load_config()
    cfg['cascade']['classify_interval_s'] = .1
    pipe = FramePipeline(cfg, None, detector=One(), landmark_factory=Hands, classifier=classifier)
    rgb = np.zeros((200, 200, 3), np.uint8)
    try:
        for t in (0., .03, .06, .09, .12):
            pipe.process(rgb, t)
        assert len(calls) == 2                     # at t=0 and t=0.12, reused in between
    finally:
        pipe.close()


def test_detector_separate_person_and_object_thresholds_and_no_person_masks():
    class T:
        def __init__(self, v): self.v = np.asarray(v)
        def cpu(self): return self
        def numpy(self): return self.v
        def tolist(self): return self.v.tolist()
        def item(self): return self.v.item()
        def __getitem__(self, i): return T(self.v[i])

    boxes = [SimpleNamespace(xyxy=T([[0, 0, 10, 10]]), cls=T(c), conf=T(s)) for c, s in ((0, .45), (1, .4), (0, .8))]
    result = SimpleNamespace(boxes=boxes, names={0: 'person', 1: 'knife'},
                             masks=SimpleNamespace(data=T(np.ones((3, 12, 20)))))

    class Model:
        def predict(self, image, **kw):
            assert kw['conf'] == .35
            return [result]

    det = SegmentationDetector.__new__(SegmentationDetector)
    det.model, det.confidence, det.object_confidence, det.error = Model(), .5, .35, ''
    out = det.detect(np.zeros((12, 20, 3), np.uint8))
    assert [(d.label, d.confidence) for d in out] == [('knife', .4), ('person', .8)]
    assert out[0].mask is not None and out[1].mask is None


# ---- sensitivity: brief dropouts and moving thin objects must not restart the alert history ----

class _Face:
    def detect(self, rgb, t):
        face = np.zeros((478, 3))
        face[234, :2] = [20, 40]; face[454, :2] = [100, 40]
        face[[78, 13, 308, 14], :2] = [[50, 50], [60, 45], [70, 50], [60, 55]]
        face[[33, 133], :2] = [40, 30]; face[[362, 263], :2] = [80, 30]
        hand = np.column_stack((np.linspace(50, 70, 21), np.linspace(50, 65, 21), np.zeros(21)))
        return SimpleNamespace(t=t, width=rgb.shape[1], height=rgb.shape[0], face=face,
                               hands=[hand], face_matrix=np.eye(4))

    def close(self):
        pass


def _unsafe(*a):
    return dict(p_unsafe_raw=.99, p_unsafe_cal=.99, obj_conf=.99, obj_class='knife', p_unknown_cal=0., top3=[])


def _first_danger(cfg, object_at):
    """object_at(frame index) -> x offset of a 15x15 knife mask, or None for a missed detection."""
    class Det:
        ok = True
        i = -1
        def detect(self, rgb):
            self.i += 1
            out = [Detection((0, 0, 160, 160), 'person', .99)]
            x = object_at(self.i)
            if x is not None:
                mask = np.zeros(rgb.shape[:2], bool); mask[45:60, x:x + 15] = True
                out.append(Detection((x, 45, x + 15, 60), 'knife', .99, mask))
            return out
    cfg['quality']['min_sharpness'] = 0
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=_Face, classifier=_unsafe)
    rgb = np.full((160, 160, 3), 140, np.uint8)
    try:
        for i in range(30):
            _, _, out = pipe.process(rgb, i * .1)
            if out['level'] == 2:
                return round(i * .1, 1)
        return None
    finally:
        pipe.close()


def test_single_missed_detection_pauses_instead_of_restarting_dwell():
    flicker = lambda i: None if i == 4 else 65
    with_grace = _first_danger(load_config(), flicker)
    cfg = load_config(); cfg['temporal']['dropout_grace_s'] = 0.
    without = _first_danger(cfg, flicker)
    assert with_grace is not None and (without is None or with_grace < without)
    assert _first_danger(load_config(), lambda i: None if 4 <= i <= 7 else 65) > with_grace  # long gaps still reset


def test_moving_thin_object_keeps_its_history():
    wobble = lambda i: 65 if i % 2 == 0 else 78      # 13 px jumps: IoU < 0.2 between frames
    assert _first_danger(load_config(), wobble) is not None
