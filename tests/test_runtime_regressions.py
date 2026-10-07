import math

from childsafe.alerts.processor import AlertProcessor, DANGER
from childsafe.config import load_config


def feature(t, visible=True):
    return dict(t=t, face_ok=visible, hand_ok=True,
                d_face_width=0.1 if visible else math.nan, d_last_mouth=0.1)


def test_category_uncertainty_overrides_binary_probability():
    p = AlertProcessor(load_config())
    out = p.step(feature(0), dict(p_unsafe_raw=.95, p_unsafe_cal=.95,
                                 obj_conf=.3, obj_class='knife'))
    assert out['obj_status'] == 'uncertain'


def test_held_mouth_does_not_complete_danger_dwell():
    p = AlertProcessor(load_config())
    obj = dict(p_unsafe_raw=.95, p_unsafe_cal=.95, obj_class='knife')
    for i in range(7):
        p.step(feature(i / 30), obj)
    for i in range(7, 29):
        assert p.step(feature(i / 30, False), obj)['level'] != DANGER


def test_cooldown_keeps_repeat_danger_visible():
    p = AlertProcessor(load_config())
    obj = dict(p_unsafe_raw=.95, p_unsafe_cal=.95, obj_class='knife')
    outs = []
    for i in range(180):
        f = feature(i / 30)
        if 60 <= i < 90:
            f['d_face_width'] = 1.
        outs.append(p.step(f, obj))
    assert any(o['notify'] for o in outs[:60])
    assert any(o['level'] == DANGER for o in outs[90:])
    assert not any(o['notify'] for o in outs[90:])
