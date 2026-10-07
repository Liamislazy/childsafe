import csv
import math

from childsafe.alerts.event_log import EventLogger
from childsafe.alerts.processor import AlertProcessor, DANGER
from childsafe.config import load_config


def f(t, **overrides):
    base = dict(t=t,schema_version=2,face_ok=True,hand_ok=True,mouth_visible=True,
                detector_ok=True,classifier_ok=True,calibrated=True,object_coverage=True,
                proximity_evidence=True,object_d_relative=.1,d_face_width=.1)
    base.update(overrides)
    return base


OBJ = dict(p_unsafe_raw=.9,p_unsafe_cal=.9,obj_conf=.9,obj_class='knife')


def dangerous_processor():
    cfg = load_config(); cfg['smoothing']['method'] = 'none'
    p = AlertProcessor(cfg)
    for i in range(30):
        out = p.step(f(i/30),OBJ)
    assert out['level'] == DANGER
    return p


def test_source_disconnect_and_person_departure_end_immediately():
    for flags,reason in [(dict(source_ok=False),'source_disconnected'),
                         (dict(person_left=True),'person_left_frame')]:
        p = dangerous_processor()
        out = p.step(f(1,mouth_visible=False,**flags))
        assert out['end_reason'] == reason and not out['held']


def test_observed_departure_is_distinct_from_loss():
    p = dangerous_processor()
    out = p.step(f(1,object_d_relative=2,d_face_width=2),OBJ)
    assert out['end_reason'] == 'observed_departure'


def test_missing_mask_cannot_preload_confirmed_dwell():
    cfg = load_config(); cfg['smoothing']['method'] = 'none'
    p = AlertProcessor(cfg)
    for i in range(90):
        assert p.step(f(i/30,proximity_evidence=False),OBJ)['level'] != DANGER
    assert p.step(f(3),OBJ)['level'] != DANGER
    assert p.step(f(3.1),OBJ)['dwell'] <= .1+1e-8


def test_suspected_hand_proximity_does_not_preload_mask_time_window():
    cfg = load_config(); cfg['smoothing']['method'] = 'none'; cfg['rules']['dwell_danger_s'] = 0
    p = AlertProcessor(cfg)
    for i in range(30):
        p.step(f(i/30,proximity_evidence=False),OBJ)
    assert p.step(f(1),OBJ)['level'] != DANGER
    assert p.step(f(1.1),OBJ)['level'] == DANGER


def test_missing_object_mask_holds_risk_without_accumulating_evidence():
    p = dangerous_processor()
    out = p.step(f(1,proximity_evidence=False,object_present=False))
    assert out['level'] == DANGER and out['held'] and out['dwell'] == 0
    out = p.step(f(2,proximity_evidence=False,object_present=False))
    assert out['level'] != DANGER and out['end_reason'] == 'observation_lost'


def test_model_failure_and_ambiguous_identity_clear_cached_classification():
    p = dangerous_processor()
    out = p.step(f(1,classifier_ok=False),OBJ)
    assert out['level'] != DANGER and out['obj_status'] == 'none'
    p = dangerous_processor()
    out = p.step(f(1,association_ambiguous=True,history_reset=True),OBJ)
    assert out['level'] != DANGER and not out['green']


def test_event_log_keeps_person_and_termination_reason(tmp_path):
    logger = EventLogger(tmp_path)
    for pid in [1,2]:
        logger.update(dict(t=0,person_id=pid,level=DANGER,reason='visible proximity',p_unsafe=.9,obj_class='knife'))
    logger.update(dict(t=1,person_id=1,level=0,reason='',p_unsafe=math.nan,obj_class='',end_reason='observation_lost'))
    logger.close(2,reason='source_disconnected')
    with logger.path.open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert [(r['person_id'],r['end_reason']) for r in rows] == [('1','observation_lost'),('2','source_disconnected')]
