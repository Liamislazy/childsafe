import pandas as pd
from childsafe.config import load_config
from childsafe.evaluation.report import recording_report


def test_missed_person_and_unobserved_events_remain_in_total_recall():
    feats = pd.DataFrame([dict(t=i/10,schema_version=2,person_id=1,session_id=1,
                              mouth_visible=False,face_ok=False,hand_ok=False) for i in range(31)])
    gt = pd.DataFrame(dict(start=[1.,1.],end=[2.,2.],label=['unsafe_mouthing']*2,person_id=[1,2]))
    report = recording_report(feats,gt,load_config())
    assert report['n_gt_events'] == 2 and report['event_recall'] == 0
    assert report['observation_coverage'] == 0
    assert report['n_fully_observed_events'] == 0


def test_reconnect_requires_session_annotations():
    import pytest
    feats = pd.DataFrame([dict(t=0,schema_version=2,person_id=1,session_id=s,face_ok=False,hand_ok=False)
                          for s in [1,2]])
    gt = pd.DataFrame(dict(start=[0],end=[1],label=['unsafe_mouthing']))
    with pytest.raises(ValueError,match='session_id'):
        recording_report(feats,gt,load_config())
