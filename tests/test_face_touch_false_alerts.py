"""Bare hands must not start object warnings; real localized objects still do."""
import math

import pandas as pd
import pytest

from childsafe.alerts.processor import AlertProcessor
from childsafe.config import load_config
from childsafe.evaluation.replay import replay


def features(t, hand=.1, object_distance=math.nan, present=False, trusted=False, **kw):
    row=dict(t=t,schema_version=2,face_ok=True,hand_ok=True,mouth_visible=True,
        detector_ok=True,classifier_ok=True,calibrated=False,object_coverage=False,
        object_present=present,d_face_width=hand,object_d_relative=object_distance,
        candidate_object_d_relative=object_distance,proximity_evidence=trusted,
        evidence_source='mask' if trusted else 'hand_crop' if not present else '',
        object_label='knife' if present else '',object_conf=.99 if present else math.nan)
    row.update(kw)
    return row


def config():
    cfg=load_config(); cfg['smoothing']['method']='none'
    return cfg


def crop(label='other'):
    return dict(obj_class=label,obj_conf=.99,p_unsafe_raw=.99 if label=='knife' else .01,
                p_unsafe_cal=.99 if label=='knife' else .01,p_unknown_cal=.99 if label=='other' else 0.)


@pytest.mark.parametrize('label',['other','knife','fruit'])
@pytest.mark.parametrize('distance',[.01,.1,.3])
def test_repeated_face_touch_never_warns_even_if_hand_crop_is_misclassified(label,distance):
    cfg=config(); proc=AlertProcessor(cfg); rows=[]
    for i in range(30):
        row=features(i*.1,hand=distance); obj=crop(label)
        out=proc.step(row,obj); rows.append(dict(row,**obj))
        assert out['level']==0 and not out['notify'] and not out['green']
        assert out['risk_distance_source']=='none'
    assert replay(pd.DataFrame(rows),cfg).level.tolist()==[0]*30


def test_hand_touch_then_face_disappears_does_not_create_a_new_event():
    proc=AlertProcessor(config())
    for i in range(10): assert proc.step(features(i*.1),crop())['level']==0
    for i in range(10,25):
        out=proc.step(features(i*.1,face_ok=False,mouth_visible=False,d_last_mouth=.1),crop())
        assert out['level']==0 and not out['held']


def test_hand_close_but_localized_object_far_does_not_warn():
    proc=AlertProcessor(config())
    for i in range(25):
        out=proc.step(features(i*.1,hand=.01,present=True,object_distance=.9),crop('knife'))
        assert out['level']==0 and out['d_raw']==.9


def test_long_object_near_mouth_warns_even_with_hand_far():
    proc=AlertProcessor(config())
    outputs=[proc.step(features(i*.1,hand=1.5,present=True,object_distance=.05,trusted=True),crop())
             for i in range(20)]
    assert outputs[-1]['level']==2 and outputs[-1]['hazard_source']=='detector'


def test_box_candidate_needs_persistence_and_is_never_danger():
    proc=AlertProcessor(config())
    results=[proc.step(features(i*.1,present=True,object_distance=.1),crop('knife')) for i in range(20)]
    assert [r['level'] for r in results[:3]]==[0,0,1]
    assert max(r['level'] for r in results)==1


def test_a_single_false_object_detection_followed_by_bare_hands_does_not_warn():
    proc=AlertProcessor(config())
    assert proc.step(features(0,present=True,object_distance=.1),crop('knife'))['level']==0
    assert all(proc.step(features(i*.1),crop('knife'))['level']==0 for i in range(1,20))


def test_one_candidate_glimpse_followed_by_hidden_mouth_does_not_start_an_alert():
    proc=AlertProcessor(config())
    assert proc.step(features(0,present=True,object_distance=.1))['level']==0
    for i in range(1,20):
        out=proc.step(features(i*.1,face_ok=False,mouth_visible=False,d_last_mouth=.1))
        assert out['level']==0 and not out['held']


def test_after_real_object_loss_hand_at_face_cannot_renew_caution():
    cfg=config(); proc=AlertProcessor(cfg); rows=[]; live=[]
    for i in range(5):
        row=features(i*.1,present=True,object_distance=.1)
        out=proc.step(row); rows.append(row); live.append(out['level'])
    assert out['level']==1
    last_evidence=rows[-1]['t']
    after=[]
    for i in range(5,30):
        row=features(i*.1,face_ok=False,mouth_visible=False,d_last_mouth=.1)
        out=proc.step(row); rows.append(row); after.append(out); live.append(out['level'])
    assert after[0]['level']==1 and after[-1]['level']==0
    assert all(r['level']==0 for r in after if r['t']-last_evidence>cfg['smoothing']['lost_hold_s']+1e-8)
    assert any(r['end_reason']=='observation_lost' for r in after)
    assert replay(pd.DataFrame(rows),cfg).level.tolist()==live


def test_csv_mixed_integer_boolean_flags_replay_like_live():
    import io
    cfg=config(); proc=AlertProcessor(cfg); rows=[]; live=[]
    for i in range(20):
        row=features(i*.1,present=True,trusted=True,object_distance=.05)
        # Matches GeometryExtractor's 1 vs held-row False in the live CSV.
        row['hand_ok'] = 1 if i<19 else False
        out=proc.step(row); rows.append(row); live.append(out['level'])
    data=pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)))
    assert isinstance(data.hand_ok.iloc[0],str)
    assert replay(data,cfg).level.tolist()==live
    assert 2 in live
