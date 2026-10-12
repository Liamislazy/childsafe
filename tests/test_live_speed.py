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
        def detect(self, rgb):   # IoU 0.57 (crossing) -> tracker marks both ambiguous -> new IDs every frame
            return [Detection((10, 10, 120, 200), 'person', .9), Detection((40, 10, 150, 200), 'person', .9)]

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
    cfg['rules']['hand_crop_fallback'] = False   # isolate the mask path; the fallback is tested separately
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


def _levels_without_yolo_object(cfg, result):
    class PersonOnly:
        ok = True
        def detect(self, rgb):
            return [Detection((0, 0, 160, 160), 'person', .99)]
    cfg['quality']['min_sharpness'] = 0
    pipe = FramePipeline(cfg, None, detector=PersonOnly(), landmark_factory=_Face, classifier=lambda *a: dict(result))
    try:
        return [pipe.process(np.full((160, 160, 3), 140, np.uint8), i * .1)[2]['level'] for i in range(20)]
    finally:
        pipe.close()


def test_hand_crop_fallback_when_yolo_misses_the_held_object():
    unsafe = dict(p_unsafe_raw=.97, p_unsafe_cal=.95, obj_conf=.95, obj_class='knife', p_unknown_cal=0., top3=[])
    assert 2 in _levels_without_yolo_object(load_config(), unsafe)                    # confident unsafe -> DANGER
    unsure = dict(unsafe, p_unsafe_cal=.7, obj_conf=.75)
    assert 2 not in _levels_without_yolo_object(load_config(), unsure)                # below 0.8 -> no DANGER
    other = dict(unsafe, p_unsafe_cal=.05, obj_conf=.9, obj_class='other', p_unknown_cal=.9)
    levels = _levels_without_yolo_object(load_config(), other)
    assert 2 not in levels and 1 in levels                                            # pen/phone -> CAUTION only
    cfg = load_config(); cfg['rules']['hand_crop_fallback'] = False
    assert 2 not in _levels_without_yolo_object(cfg, unsafe)                          # switch off -> old behaviour


# ---- distance is object -> mouth whenever an object is known ----

def _run_with_object(hand_at, obj_box, conf, use_mask=True, frames=12):
    class Face(_Face):
        def detect(self, rgb, t):
            out = super().detect(rgb, t)
            x, y = hand_at
            out.hands = [np.column_stack((np.linspace(x, x + 20, 21), np.linspace(y, y + 15, 21), np.zeros(21)))]
            return out

    class Det:
        ok = True
        def detect(self, rgb):
            x0, y0, x1, y1 = obj_box
            mask = None
            if use_mask:
                mask = np.zeros(rgb.shape[:2], bool); mask[y0:y1, x0:x1] = True
            return [Detection((0, 0, 160, 160), 'person', .99), Detection(obj_box, 'fork', conf, mask)]

    cfg = load_config(); cfg['quality']['min_sharpness'] = 0; cfg['smoothing']['method'] = 'none'
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=Face, classifier=_unsafe)
    try:
        outs = [pipe.process(np.full((160, 160, 3), 140, np.uint8), i * .1) for i in range(frames)]
        return outs[-1][2]['persons'][0], outs[-1][2]['feature_rows'][0], [o[2]['level'] for o in outs]
    finally:
        pipe.close()


def test_object_far_from_mouth_while_hand_is_near_is_not_near():
    # hand right at the mouth, but the (low-confidence) object it holds points away, 60+ px down-right
    person, row, levels = _run_with_object(hand_at=(55, 55), obj_box=(100, 110, 140, 150), conf=.45)
    assert row['distance_source'] == 'object'
    assert person['d'] > 0.5 and not person['near_frame']
    assert max(levels) == 0


def test_long_object_tip_at_mouth_while_hand_is_far_is_near():
    # hand ~0.7 face widths away, the object it holds reaches the mouth; low confidence -> CAUTION only
    person, row, levels = _run_with_object(hand_at=(105, 100), obj_box=(60, 50, 110, 105), conf=.45)
    assert row['distance_source'] == 'object' and person['d'] < 0.1
    assert 1 in levels and 2 not in levels


def test_object_without_mask_uses_its_box_and_never_confirms_danger():
    person, row, levels = _run_with_object(hand_at=(105, 100), obj_box=(60, 50, 110, 105), conf=.9, use_mask=False)
    assert row['distance_source'] == 'object' and person['d'] < 0.1
    assert not row['proximity_evidence'] and 2 not in levels
