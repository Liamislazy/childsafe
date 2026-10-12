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
