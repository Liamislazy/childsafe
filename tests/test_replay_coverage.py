import math
import pandas as pd
from childsafe.alerts.processor import AlertProcessor, DANGER
from childsafe.config import load_config
from childsafe.evaluation.replay import replay, aggregate_results
from childsafe.evaluation.coverage import observation_metrics


def row(t,pid=1,session=1,visible=True):
    return dict(t=t,person_id=pid,session_id=session,schema_version=2,mouth_visible=visible,
                face_ok=visible,hand_ok=True,detector_ok=True,classifier_ok=True,calibrated=True,
                object_coverage=True,quality_reasons=[],proximity_evidence=visible,object_d_relative=.1,
                d_face_width=.1,p_unsafe_raw=.9,p_unsafe_cal=.9,obj_conf=.9,obj_class='knife')


def test_replay_matches_processor_per_person_and_session():
    cfg = load_config(); cfg['smoothing']['method'] = 'none'
    rows = []
    for session in [1,2]:
        for i in range(40):
            for pid in [1,2]:
                f = row(i/30,pid,session,visible=pid == 1)
                rows.append(f)
    out = replay(pd.DataFrame(rows),cfg)
    expected = []
    for session in [1,2]:
        processors = {pid:AlertProcessor(cfg) for pid in [1,2]}
        for f in [x for x in rows if x['session_id'] == session]:
            expected.append(processors[f['person_id']].step(f,f)['level'])
    assert out.level.tolist() == expected
    assert not (out[out.person_id == 2].level == DANGER).any()
    assert (out.groupby('session_id').first().level == 0).all()


def test_coverage_counts_lost_intervals_and_does_not_double_count_people():
    cfg = load_config()
    rows = [row(t,pid,visible=t <= .2) for t in [0,.1,.2,.3,1.] for pid in [1,2]]
    out = replay(pd.DataFrame(rows),cfg)
    metrics = observation_metrics(out,cfg)
    assert metrics['observation_coverage'] == .2
    assert metrics['duration_s'] == 1.
    assert len(aggregate_results(out)) == 5


def test_legacy_coverage_is_unknown():
    cfg = load_config()
    out = replay(pd.DataFrame([dict(t=0,face_ok=1,hand_ok=1,d_face_width=.1)]),cfg)
    metrics = observation_metrics(out,cfg)
    assert math.isnan(metrics['observation_coverage'])
    assert metrics['observation_note'] == 'không có dữ liệu chất lượng quan sát'


def test_category_confidence_survives_csv_roundtrip(tmp_path):
    rows = [row(i/30) for i in range(60)]
    for f in rows: f['obj_conf'] = .2
    path = tmp_path/'features.csv'
    pd.DataFrame(rows).to_csv(path,index=False)
    out = replay(pd.read_csv(path),load_config())
    assert (out.obj_status == 'uncertain').all()
    assert not (out.level == DANGER).any()
