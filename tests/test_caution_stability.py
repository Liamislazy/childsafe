"""CAUTION must not flicker while an object stays at (and covers) the mouth."""
import math

from childsafe.alerts.processor import AlertProcessor, CAUTION, NORMAL
from childsafe.config import load_config


def f(t, near=True, **kw):
    d = .1 if near else .9
    base = dict(t=t, schema_version=2, face_ok=True, hand_ok=True, mouth_visible=True, detector_ok=True,
                classifier_ok=True, object_present=True, proximity_evidence=False,
                candidate_object_d_relative=d, d_face_width=d)
    base.update(kw)
    return base


def proc(hold=1.5):
    cfg = load_config(); cfg['smoothing']['method'] = 'none'; cfg['rules']['caution_hold_s'] = hold
    return AlertProcessor(cfg)


def covered(t, hand_at_old_mouth=True):
    # object in the mouth hides it: no usable mouth, face often lost, hand where the mouth was
    return f(t, mouth_visible=False, face_ok=False, d_face_width=math.nan,
             candidate_object_d_relative=math.nan, d_last_mouth=.1 if hand_at_old_mouth else 1.5)


def levels(p, frames):
    return [p.step(x)['level'] for x in frames]


def rejected(t):
    # face still found, but the mouth is rejected this frame (blur / jumping landmarks as the object covers it)
    return f(t, mouth_visible=False, quality_reasons=['unstable mouth landmarks'])


def test_flickering_evidence_keeps_caution_steady():
    frames = [f(i * .1) if i % 2 == 0 else rejected(i * .1) for i in range(30)]   # every other frame lost
    lv = levels(proc(), frames)
    assert all(l >= CAUTION for l in lv[1:])
    assert NORMAL in levels(proc(hold=0), frames)                                  # old behaviour flickers


def test_object_kept_in_covered_mouth_stays_caution_beyond_the_hold():
    p = proc()
    levels(p, [f(i * .1) for i in range(5)])
    lv = levels(p, [covered(.5 + i * .1) for i in range(60)])                       # 6 s with the mouth covered
    assert all(l == CAUTION for l in lv)


def test_hand_moving_away_while_mouth_hidden_ends_after_the_hold():
    p = proc()
    levels(p, [f(i * .1) for i in range(5)])
    lv = levels(p, [covered(.5 + i * .1, hand_at_old_mouth=False) for i in range(30)])
    assert lv[0] == CAUTION and lv[-1] == NORMAL                                    # ends ~1.5 s later


def test_observed_departure_ends_caution_immediately():
    p = proc()
    levels(p, [f(i * .1) for i in range(5)])
    assert p.step(f(.5, near=False))['level'] == NORMAL


def test_hold_never_starts_without_real_evidence():
    assert set(levels(proc(), [covered(i * .1) for i in range(20)])) == {NORMAL}
