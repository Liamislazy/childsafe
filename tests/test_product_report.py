import pandas as pd
import pytest
from childsafe.config import load_config
from childsafe.evaluation.product_report import evaluate_session,summarize


def features():
    return pd.DataFrame([dict(t=t,session_id=1,person_id=1,schema_version=2,source_ok=True,
                             frame_valid=True,mouth_visible=False,detector_ok=True,classifier_ok=True,
                             d_face_width=.1,face_ok=False,hand_ok=False) for t in [0,.1,.2,.3]])


def test_unknown_exposure_not_negative_and_missing_person_counts_as_missed():
    gt=pd.DataFrame([dict(start=.1,end=.25,label='unsafe_mouthing',person_id=99)])
    r=evaluate_session(features(),gt,load_config(),0,1,'person')
    assert r['events']==1 and r['event_recall']==0 and r['negative_hours']==0
    assert r['false_alarms_per_hour'] is None and r['observation_coverage']==0


def test_zero_false_alarms_still_has_nonzero_confidence_upper_bound():
    gt=pd.DataFrame([dict(start=0,end=1,label='non_event')])
    r=evaluate_session(features(),gt,load_config(),0,1)
    assert r['false_alarms']==0 and r['fa_poisson_95'][1]>0
    with pytest.raises(ValueError): evaluate_session(features(),gt.assign(end=2),load_config(),0,1)


def test_positive_and_negative_alarm_overlap_cannot_hide_false_alarms():
    gt=pd.DataFrame([dict(start=0,end=.15,label='unsafe_mouthing'),dict(start=.15,end=.4,label='non_event')])
    r=evaluate_session(features(),gt,load_config(),0,1,baseline=True)
    assert r['hits']==1 and r['false_alarms']==1
