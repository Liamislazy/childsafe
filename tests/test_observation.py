from childsafe.alerts.observation import observation
from childsafe.alerts.processor import AlertProcessor, DANGER
from childsafe.config import load_config


def f(t=0, **kw):
    return dict(t=t, schema_version=2, face_ok=True, hand_ok=True, mouth_visible=True,
                detector_ok=True, classifier_ok=True, calibrated=True, object_coverage=True,
                d_face_width=.1, **kw)


def test_missing_mouth_and_model_are_not_green():
    x = f(); x['mouth_visible'] = False
    assert observation(x)[0] == 'unobservable'
    x = f(); x['detector_ok'] = False
    p = AlertProcessor(load_config())
    assert not p.step(x)['green']


def test_legacy_has_unknown_coverage():
    assert observation({}) == ('degraded', ['legacy: no observation quality data'])


def test_short_occlusion_preserves_danger_long_loss_ends_event():
    p = AlertProcessor(load_config())
    obj = dict(p_unsafe_raw=.9, p_unsafe_cal=.9, obj_class='knife')
    for i in range(60):
        out = p.step(f(i/30), obj)
    assert out['level'] == DANGER
    x = f(2.1); x['mouth_visible'] = False
    assert p.step(x)['level'] == DANGER
    x['t'] = 3
    out = p.step(x)
    assert out['end_reason'] == 'observation_lost'
    assert not out['green']
