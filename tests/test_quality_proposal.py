import pandas as pd
import pytest
from childsafe.config import load_config
from childsafe.evaluation.quality_calibration import propose_quality


def test_proposal_dev_only_keeps_calibration_unverified():
    f = pd.DataFrame([dict(session_id=1,person_id=1,t=t,quality_schema_version=3,
                           original_face_px=face,original_mouth_px=10,head_yaw=0,head_pitch=0,
                           brightness=120,sharpness=50) for t,face in [(0,60),(1,20)]])
    labels = f[['session_id','person_id','t']].copy(); labels['mouth_observable'] = [1,0]
    cfg, report = propose_quality(f,labels,load_config(),'dev')
    assert report['selected']['false_visible'] == 0 and not cfg['quality']['calibrated']
    with pytest.raises(ValueError): propose_quality(f,labels,load_config(),'test')
    with pytest.raises(ValueError): propose_quality(f,labels.assign(mouth_observable=1),load_config(),'dev')
