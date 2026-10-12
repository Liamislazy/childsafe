"""One real person must keep one ID and one card; real second people must stay separate."""
import numpy as np

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection
from childsafe.vision.tracking import PersonTracker, dedupe_people
from childsafe.web.app import _displayed_people

BODY = (100, 50, 300, 470)


def person(box, conf=.9):
    return Detection(box, 'person', conf)


def run(tracker, frames, dt=.15):
    ids, amb = [], []
    for i, dets in enumerate(frames):
        active, _ = tracker.update(dets, i * dt)
        ids.append([t.person_id for t in active]); amb.append([t.ambiguous for t in active])
    return ids, amb


def test_duplicate_upper_body_box_is_merged_into_one_person():
    frames = [[person(BODY), person((110, 40, 290, 260), .6)]] * 20
    ids, amb = run(PersonTracker(hold_s=3), frames)
    assert all(len(f) == 1 for f in ids) and len({i for f in ids for i in f}) == 1
    assert not any(a for f in amb for a in f)


def test_real_overlapping_people_are_not_merged():
    assert len(dedupe_people([person(BODY), person((160, 200, 240, 460))])) == 2   # child/doll on a lap
    assert len(dedupe_people([person(BODY), person((110, 220, 290, 470))])) == 2   # shorter child in front
    assert len(dedupe_people([person((70, 0, 170, 200)), person((90, 0, 190, 200))])) == 2  # two crossing


def test_two_separate_people_keep_two_stable_ids():
    frames = [[person((20, 50, 200, 470)), person((420, 50, 600, 470))]] * 15
    ids, amb = run(PersonTracker(hold_s=3), frames)
    assert all(len(f) == 2 for f in ids) and len({i for f in ids for i in f}) == 2
    assert not any(a for f in amb for a in f)


def test_fast_motion_keeps_the_id():
    xs = [50 + 130 * k for k in (0, 1, 2, 3, 2, 1)] * 3   # IoU between frames ~0.2 < min_iou
    ids, _ = run(PersonTracker(hold_s=3), [[person((x, 50, x + 200, 470))] for x in xs])
    assert len({i for f in ids for i in f}) == 1


def test_one_person_is_shown_once_in_the_web_runtime():
    class Landmarks:
        def detect(self, rgb, t):
            from types import SimpleNamespace
            return SimpleNamespace(t=t, width=rgb.shape[1], height=rgb.shape[0], face=None, hands=[], face_matrix=None)
        def close(self): pass

    seq = [[person(BODY)], [], [person((230, 50, 430, 470))], [person((360, 50, 560, 470))], []] * 4

    class Det:
        ok = True
        i = -1
        def detect(self, rgb):
            self.i += 1
            return seq[self.i]

    pipe = FramePipeline(load_config(), None, detector=Det(), landmark_factory=Landmarks, classifier=lambda *a: None)
    try:
        for i in range(len(seq)):
            _, _, out = pipe.process(np.zeros((480, 640, 3), np.uint8), i * .15)
            assert len(_displayed_people(out['persons'])) <= 1
    finally:
        pipe.close()


def test_displayed_person_number_is_reused_not_ever_increasing():
    from childsafe.web.app import RuntimeDemo
    demo = RuntimeDemo.__new__(RuntimeDemo)
    shown = [demo._slot([pid])[pid] for pid in range(10, 60)]   # internal IDs keep climbing
    assert set(shown) == {1}
    slots = demo._slot([60, 61])                              # a second person gets 2
    assert sorted(slots.values()) == [1, 2]
    assert demo._slot([61, 62]) == {61: 2, 62: 1}               # freed number 1 is reused


def test_single_person_mode_ignores_a_small_second_person():
    small = person((452, 334, 576, 480), .7)        # e.g. photo/doll at the bottom right (seen in a real session)
    frames = [[person((96, 166, 564, 480)), small]] * 10
    ids, amb = run(PersonTracker(hold_s=3, primary_only=True), frames)
    assert all(len(f) == 1 for f in ids) and len({i for f in ids for i in f}) == 1 and not any(a for f in amb for a in f)
    ids, _ = run(PersonTracker(hold_s=3), frames)   # default: both are tracked
    assert all(len(f) == 2 for f in ids)


# ---- two real people side by side (geometry from a real session, 2026-10-12) ----
from types import SimpleNamespace
from childsafe.vision.tracking import resolve_hands

ME, FRIEND = (96, 166, 564, 480), (452, 334, 576, 480)       # boxes overlap: IoU ~0.11


def _face(cx, cy, w):
    face = np.zeros((478, 3)); face[:, :2] = [cx, cy]          # all points around the face centre
    face[234, :2] = [cx - w / 2, cy]; face[454, :2] = [cx + w / 2, cy]
    face[[78, 13, 308, 14], :2] = [[cx - 10, cy + 25], [cx, cy + 20], [cx + 10, cy + 25], [cx, cy + 30]]
    face[[33, 133], :2] = [cx - 30, cy - 15]; face[[362, 263], :2] = [cx + 30, cy - 15]
    return face


def _hand(x, y):
    return np.column_stack((np.linspace(x, x + 40, 21), np.linspace(y, y + 40, 21), np.zeros(21)))


def test_side_by_side_people_are_not_ambiguous_from_slight_overlap():
    ids, amb = run(PersonTracker(hold_s=3), [[person(ME), person(FRIEND, .7)]] * 10)
    assert all(len(f) == 2 for f in ids) and len({i for f in ids for i in f}) == 2
    assert not any(a for f in amb for a in f)


def test_duplicate_hand_goes_to_the_mouth_it_is_at():
    my_hand = _hand(300, 300)                                   # at my mouth (face centre 300,270)
    me = SimpleNamespace(face=_face(300, 270, 140), hands=[my_hand], handedness=['Right'])
    friend = SimpleNamespace(face=_face(515, 380, 48), hands=[my_hand.copy()], handedness=['Right'])
    amb = resolve_hands({1: me, 2: friend}, {1: ME, 2: FRIEND})
    assert amb == set() and len(me.hands) == 1 and len(friend.hands) == 0


def test_hand_equally_close_to_two_mouths_is_still_ambiguous():
    hand = _hand(380, 320)
    a = SimpleNamespace(face=_face(330, 300, 100), hands=[hand], handedness=['Left'])
    b = SimpleNamespace(face=_face(440, 300, 100), hands=[hand.copy()], handedness=['Right'])
    assert resolve_hands({1: a, 2: b}, {1: (200, 150, 470, 480), 2: (330, 150, 600, 480)}) == {1, 2}


def test_hand_at_mouth_reaches_danger_with_a_friend_next_to_me():
    class Det:
        ok = True
        def detect(self, rgb):
            return [person(ME), person(FRIEND, .7)]

    class Crops:                                                # landmarks in crop coordinates
        def detect(self, rgb, t):
            h, w = rgb.shape[:2]
            if (w, h) == (ME[2] - ME[0], ME[3] - ME[1]):
                ox, oy, face, hands = ME[0], ME[1], _face(300, 270, 140), [_hand(285, 285)]
            else:                                               # friend's crop also sees my hand
                ox, oy, face, hands = FRIEND[0], FRIEND[1], _face(515, 380, 48), [_hand(285, 285)]
            shift = np.array([ox, oy, 0])
            hands = [hd - shift for hd in hands if (hd[:, 0] - ox).min() >= -60]
            return SimpleNamespace(t=t, width=w, height=h, face=face - shift, hands=hands,
                                   handedness=['Right'] * len(hands), face_matrix=np.eye(4))
        def close(self): pass

    unsafe = lambda *a: dict(p_unsafe_raw=.97, p_unsafe_cal=.95, obj_conf=.95, obj_class='knife', p_unknown_cal=0., top3=[])
    cfg = load_config(); cfg['quality']['min_sharpness'] = 0
    pipe = FramePipeline(cfg, None, detector=Det(), landmark_factory=Crops, classifier=unsafe)
    try:
        levels = {}
        for i in range(20):
            _, _, out = pipe.process(np.full((480, 640, 3), 140, np.uint8), i * .1)
            for p in out['persons']:
                levels.setdefault(p['person_id'], []).append(p['level'])
        assert len(levels) == 2                                 # two stable people
        assert any(2 in lv for lv in levels.values())           # my hand at my mouth -> DANGER possible
    finally:
        pipe.close()
