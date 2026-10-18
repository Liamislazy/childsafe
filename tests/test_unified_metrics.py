import pandas as pd
import pytest
from childsafe.config import load_config
from childsafe.evaluation.product_report import METRIC_VERSION
from childsafe.evaluation.session_reports import SessionRecord,score,tune
from childsafe.alerts.observation import image_observation,observation
from test_observation import f


def record(tmp_path,split='dev'):
    cfg=load_config(); rows=[]
    for i in range(20):
        row=f(i*.1); row.update(p_unsafe_raw=.99,p_unsafe_cal=.99,obj_conf=.99,obj_class='knife')
        rows.append(row|dict(session_id=1,person_id=1,calibrated=False,object_coverage=False))
    ann=pd.DataFrame([dict(start=0,end=.8,label='unsafe_mouthing'),dict(start=.8,end=2,label='non_event')])
    manifest=dict(recording_id='fixture',split=split,start_s=0,end_s=2,annotation_scope='camera',condition='elevated')
    return SessionRecord(tmp_path/'manifest.json',manifest,cfg,pd.DataFrame(rows),ann)


def test_tuning_and_reporting_use_identical_explicit_exposure(tmp_path):
    r=record(tmp_path)
    cfg,result=tune([r],'dev',candidates=[(.35,3,.6,.5)])
    report=score(r,cfg)
    assert result['selected']['false_alarms_per_hour']==report['false_alarms_per_hour']
    assert result['selected']['event_recall']==report['event_recall']
    assert report['false_alarms']==1 and report['negative_hours']==1.2/3600
    assert report['metric_definition']==METRIC_VERSION


def test_image_coverage_is_independent_of_calibration_and_supported_classes(tmp_path):
    r=record(tmp_path)
    sample=r.features.iloc[0].to_dict()
    assert image_observation(sample)[0]=='observable'
    assert observation(sample)[0]=='degraded'
    report=score(r)
    assert report['observation_coverage']==0
    assert report['image_observation_coverage']==pytest.approx(1.9/2)
    assert report['image_observable_events']==1


def test_unknown_time_is_not_negative_and_test_tuning_is_rejected(tmp_path):
    r=record(tmp_path); r.annotations=r.annotations.iloc[:1]
    assert score(r)['false_alarms_per_hour'] is None
    with pytest.raises(ValueError,match='negative exposure'): tune([r],'dev',candidates=[(.35,3,.6,.5)])
    with pytest.raises(ValueError,match='dev'): tune([record(tmp_path,'test')],'dev')
    from childsafe.evaluation.cli import main
    with pytest.raises(SystemExit): main(['grid','--sessions',str(tmp_path),'--split','test'])


def test_replay_cannot_tune_detector_on_logged_features(tmp_path):
    import copy
    r=record(tmp_path); changed=copy.deepcopy(r.cfg); changed['detector']['imgsz']=320
    with pytest.raises(ValueError,match='new sessions'): score(r,changed)
