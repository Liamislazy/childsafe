import math

import numpy as np
import pandas as pd

from childsafe.alerts.processor import CAUTION, DANGER, NORMAL, AlertProcessor, Components
from childsafe.alerts.temporal import Hysteresis, NofM, frames_to_events
from childsafe.config import load_config
from childsafe.evaluation.events import distance_only_levels, event_metrics
from childsafe.evaluation.replay import replay
from childsafe.vision.smoothing import OneEuro

FPS = 30


def cfg():
    c = load_config()
    c["smoothing"]["method"] = "none"
    return c


def feats(d, t0=0.0, face=True, hand=True, d_last=math.nan):
    return {"t": t0, "face_ok": int(face), "hand_ok": int(hand),
            "d_face_width": d if face else math.nan, "d_last_mouth": d_last,
            "p_unsafe_raw": math.nan, "p_unsafe_cal": math.nan, "obj_class": ""}


UNSAFE = {"p_unsafe_raw": 0.95, "p_unsafe_cal": 0.9, "obj_class": "knife"}
SAFE = {"p_unsafe_raw": 0.05, "p_unsafe_cal": 0.1, "obj_class": "spoon"}


def run(proc, ds, obj=None, start=0.0, **kw):
    outs = []
    for i, d in enumerate(ds):
        f = feats(d, start + i / FPS, **kw)
        o = obj if (obj and proc.wants_classification(f)) else None
        outs.append(proc.step(f, o))
    return outs


def test_hysteresis_and_nofm():
    h = Hysteresis(0.3, 0.5)
    assert [h(x) for x in (0.6, 0.29, 0.45, 0.49, 0.51, 0.4)] == [False, True, True, True, False, False]
    v = NofM(3, 5)
    assert [v(x) for x in (1, 0, 1, 0, 1, 0, 0, 0)] == [False, False, False, False, True, False, False, False]


def test_frames_to_events_merge():
    t = np.arange(10) * 1.0
    a = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0], bool)
    assert frames_to_events(t, a, 0) == [(0, 1), (3, 3), (7, 8)]
    assert frames_to_events(t, a, 2.0) == [(0, 3), (7, 8)]


def test_unsafe_object_at_mouth_reaches_danger_after_dwell():
    p = AlertProcessor(cfg())
    outs = run(p, [1.0] * 10 + [0.1] * 60, UNSAFE)
    levels = [o["level"] for o in outs]
    assert levels[0] == NORMAL
    assert DANGER in levels
    first = levels.index(DANGER)
    # needs N-of-M confirmation plus 0.5 s dwell
    assert outs[first]["t"] - outs[10]["t"] >= 0.5


def test_safe_object_caps_at_caution():
    p = AlertProcessor(cfg())
    levels = [o["level"] for o in run(p, [0.1] * 200, SAFE)]
    assert DANGER not in levels
    assert levels[30] == NORMAL and levels[-1] == CAUTION  # caution only after 3 s dwell


def test_without_object_gate_everything_near_is_danger():
    p = AlertProcessor(cfg(), Components.from_off(["object_gate"]))
    levels = [o["level"] for o in run(p, [0.1] * 60, SAFE)]
    assert DANGER in levels


def test_lost_mouth_near_hand_is_caution():
    p = AlertProcessor(cfg())
    run(p, [0.2] * 30, SAFE)
    out = p.step(feats(math.nan, 1.05, face=False, hand=True, d_last=0.2))
    assert out["lost_near"] and out["level"] >= CAUTION
    p2 = AlertProcessor(cfg(), Components.from_off(["lost_landmark"]))
    run(p2, [0.2] * 30, SAFE)
    out2 = p2.step(feats(math.nan, 1.05, face=False, hand=True, d_last=0.2))
    assert not out2["lost_near"]


def test_cooldown_suppresses_notification_not_risk():
    p = AlertProcessor(cfg())
    seq = [0.1] * 60 + [1.0] * 30 + [0.1] * 60     # danger, leave for 1 s, come back
    levels = [o["level"] for o in run(p, seq, UNSAFE)]
    assert DANGER in levels[:60]
    assert DANGER in levels[90:]
    p2 = AlertProcessor(cfg(), Components.from_off(["cooldown"]))
    levels2 = [o["level"] for o in run(p2, seq, UNSAFE)]
    assert DANGER in levels2[90:]


def test_one_euro_smooths_jitter():
    f = OneEuro(1.0, 0.0)
    rng = np.random.default_rng(0)
    x = 0.5 + rng.normal(0, 0.05, 300)
    y = [f(v, i / FPS) for i, v in enumerate(x)]
    assert np.std(y[50:]) < 0.5 * np.std(x[50:])


def test_event_metrics_and_baseline():
    t = np.arange(0, 600, 1 / FPS)
    d = np.full_like(t, 1.0)
    d[(t > 100) & (t < 104)] = 0.1        # dangerous event
    d[(t > 300) & (t < 300.2)] = 0.1      # brief flicker (baseline false alarm)
    gt = pd.DataFrame({"video": "v", "start": [100.0, 400.0], "end": [104.0, 410.0],
                       "label": ["unsafe_mouthing", "eating"]})
    lv = distance_only_levels(d, 0.35)
    m = event_metrics(t, lv, gt, 600.0, min_level=DANGER, tol=2.0, merge_s=0.0)
    assert m["event_recall"] == 1.0 and m["false_alarms"] == 1
    assert abs(m["false_alarms_per_hour"] - 1 / ((600 - 4) / 3600)) < 1e-6
    assert 0 <= m["latency_median_s"] < 0.1
    assert m["per_second_agreement"] > 0.98


def test_replay_matches_live_and_filters_flicker():
    c = cfg()
    t = np.arange(0, 60, 1 / FPS)
    d = np.full_like(t, 1.0)
    d[(t > 10) & (t < 13)] = 0.1
    d[(t > 30) & (t < 30.1)] = 0.1      # 3-frame flicker
    df = pd.DataFrame([feats(x, ti) | UNSAFE for x, ti in zip(d, t)])
    out = replay(df, c)
    ev = frames_to_events(out.t, out.level >= DANGER, 2.0)
    assert len(ev) == 1 and 10 < ev[0][0] < 11.5
