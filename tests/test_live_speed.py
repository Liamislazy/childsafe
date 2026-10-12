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

_UNSURE = dict(p_unsafe_raw=.7, p_unsafe_cal=.7, obj_conf=.75, obj_class='knife', p_unknown_cal=0., top3=[])


def _run_with_object(hand_at, obj_box, conf, use_mask=True, frames=12, result=None):
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
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=Face,
                         classifier=lambda *a: dict(result or _UNSURE))
    try:
        outs = [pipe.process(np.full((160, 160, 3), 140, np.uint8), i * .1) for i in range(frames)]
        return outs[-1][2]['persons'][0], outs[-1][2]['feature_rows'][0], [o[2]['level'] for o in outs]
    finally:
        pipe.close()


def test_both_distances_are_reported_and_the_closer_one_counts():
    # hand at the mouth, object pointing away: near through the hand
    person, row, levels = _run_with_object(hand_at=(55, 55), obj_box=(100, 110, 140, 150), conf=.45)
    assert row['d_hand'] < .2 and row['d_object'] > .5 and row['distance_source'] == 'hand'
    assert person['near_frame'] and 1 in levels and 2 not in levels
    # hand further away, object tip at the mouth: near through the object
    person, row, levels = _run_with_object(hand_at=(105, 100), obj_box=(60, 50, 110, 105), conf=.45)
    assert row['d_object'] < .1 and row['distance_source'] == 'object' and person['d'] < .1
    assert 1 in levels and 2 not in levels


def test_weak_outline_alone_never_confirms_danger():
    for use_mask in (True, False):          # low-confidence mask, or box only
        person, row, levels = _run_with_object(hand_at=(105, 100), obj_box=(60, 50, 110, 105), conf=.45,
                                               use_mask=use_mask)
        assert not row['proximity_evidence'] and 2 not in levels


def test_confident_unsafe_object_near_mouth_reaches_danger_by_either_distance():
    for hand_at, box in [((105, 100), (60, 50, 110, 105)), ((55, 55), (100, 110, 140, 150))]:
        person, row, levels = _run_with_object(hand_at=hand_at, obj_box=box, conf=.45, result=_unsafe())
        assert row['evidence_source'] == 'classifier' and 2 in levels


def test_zoomed_pass_finds_small_object_and_maps_it_back():
    class SmallFace(_Face):                     # person far away in a 640x480 frame
        def detect(self, rgb, t):
            out = super().detect(rgb, t)
            out.face[:, :2] = out.face[:, :2] * .5 + [400, 300]
            out.hands = [hd * [.5, .5, 1] + [400, 300, 0] for hd in out.hands]
            out.width, out.height = rgb.shape[1], rgb.shape[0]
            return out

    seen = []

    class Det:
        ok, supports_roi = True, True
        def detect(self, rgb):
            seen.append(rgb.shape[:2])
            if rgb.shape[:2] == (480, 640):     # full frame: the small fork is missed
                return [Detection((380, 260, 470, 380), 'person', .95)]
            mask = np.zeros(rgb.shape[:2], bool); mask[60:90, 70:80] = True   # in crop coordinates
            return [Detection((70, 60, 80, 90), 'fork', .8, mask)]

    cfg = load_config(); cfg['quality']['min_sharpness'] = 0; cfg['quality']['min_face_px'] = 10
    cfg['landmarks']['zoom'] = False      # this test is about the object pass (landmark zoom tested separately)
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=SmallFace, classifier=_unsafe)
    try:
        scene, row, out = pipe.process(np.full((480, 640, 3), 140, np.uint8), 0.)
        assert len(seen) == 2 and seen[1][0] == seen[1][1] and seen[1][0] < 480     # second, square crop
        fork = [o for o in scene.objects if o.label == 'fork'][0]
        assert fork.mask.shape == (480, 640) and fork.mask.sum() == 300
        ys, xs = np.nonzero(fork.mask)
        assert (xs.min(), ys.min()) == (fork.box[0], fork.box[1])               # mask and box agree
        assert row['roi_objects'] == 1 and row['object_present']
    finally:
        pipe.close()
    cfg['detector']['roi_zoom'] = False
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=SmallFace, classifier=_unsafe)
    try:
        assert not [o for o in pipe.process(np.full((480, 640, 3), 140, np.uint8), 0.)[0].objects]
    finally:
        pipe.close()



def test_far_person_is_zoomed_for_landmarks_and_mapped_back():
    """A far person: MediaPipe only finds the face when the crop is enlarged (as for a ~30 px face)."""
    BOX = (300, 200, 380, 330)                  # 80x130 px person in a 640x480 frame
    inputs = []

    class ZoomOnly:
        def detect(self, rgb, t):
            h, w = rgb.shape[:2]
            inputs.append((w, h))
            face = None
            hands = []
            if max(w, h) >= 400 and (w, h) != (640, 480):   # only visible after enlargement (full frame: missed)
                s = w / 104                         # crop is the 80x130 box + margins = 104 x 156 px
                cx, cy = (340 - 288) * s, (240 - 187) * s     # face centre in crop pixels
                face = np.zeros((478, 3)); face[:, :2] = [cx, cy]
                face[234, :2] = [cx - 14 * s, cy]; face[454, :2] = [cx + 14 * s, cy]
                face[[78, 13, 308, 14], :2] = [[cx - 4 * s, cy + 8 * s], [cx, cy + 6 * s], [cx + 4 * s, cy + 8 * s], [cx, cy + 10 * s]]
                face[[33, 133], :2] = [cx - 6 * s, cy - 4 * s]; face[[362, 263], :2] = [cx + 6 * s, cy - 4 * s]
            return SimpleNamespace(t=t, width=w, height=h, face=face, hands=hands, handedness=[], face_matrix=np.eye(4))
        def close(self): pass

    class Det:
        ok = True
        def detect(self, rgb):
            return [Detection(BOX, 'person', .9)]

    cfg = load_config(); cfg['quality']['min_sharpness'] = 0
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=ZoomOnly, classifier=lambda *a: None)
    try:
        scene, row, out = pipe.process(np.full((480, 640, 3), 140, np.uint8), 0.)
        assert max(inputs[-1]) >= 400                                  # MediaPipe got an enlarged crop
        lm = scene.people[out['persons'][0]['person_id']]
        assert lm.face is not None and abs(lm.face[:, 0].mean() - 340) < 2 and abs(lm.face[:, 1].mean() - 240) < 3
        assert abs(row['face_width'] - 28) < 1.5                        # 28 px in the original frame
        assert row['mouth_visible']                                     # not rejected as "face too small"
    finally:
        pipe.close()
    cfg['landmarks']['zoom'] = False
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=ZoomOnly, classifier=lambda *a: None)
    try:
        scene, row, out = pipe.process(np.full((480, 640, 3), 140, np.uint8), 0.)
        assert not row['mouth_visible']                                 # old behaviour: face missed
    finally:
        pipe.close()
