from types import SimpleNamespace
import numpy as np
import pytest
from childsafe.vision.detection import Detection
from childsafe.vision.tracking import Track,associate_object
from childsafe.vision.objects import MaskEvidence,assign_objects
from childsafe.vision.elevated import object_proximity
from test_elevated import landmarks


def object_at(offset=0):
    mask=np.zeros((100,100),bool); mask[49:52,10+offset:50+offset]=True
    return Detection((10+offset,49,50+offset,52),'knife',.99,mask)


@pytest.mark.parametrize('scale',[.5,1,2,4])
def test_assignment_is_resolution_invariant(scale):
    tracks=[Track(1,(0,0,100*scale,200*scale),0),Track(2,(200*scale,0,300*scale,200*scale),0)]
    hands={1:SimpleNamespace(hands=[np.array([[50,50,0],[70,70,0]])*scale]),
           2:SimpleNamespace(hands=[np.array([[250,50,0],[270,70,0]])*scale])}
    obj=Detection(tuple(v*scale for v in (80,50,100,70)),'knife',.99)
    assert associate_object(obj,tracks,hands)==(1,False)


def test_ambiguous_object_does_not_reset_unrelated_third_person():
    tracks=[Track(i,(0,0,100,200),0) for i in (1,2,3)]
    hands={i:SimpleNamespace(hands=[np.array([[x,50,0],[x+10,60,0]])]) for i,x in [(1,40),(2,55),(3,250)]}
    result=assign_objects([Detection((50,50,60,60),'knife',.99)],tracks,hands)
    assert result.ambiguous=={1,2} and 3 not in result.ambiguous


def test_mask_stability_requires_time_and_resets_after_gap_or_shape_jump():
    evidence=MaskEvidence({})
    lm=landmarks(); f=dict(mouth_visible=True,hand_ok=True,face_width=60)
    def step(t,obj): return evidence.apply(obj,t,object_proximity(obj,lm,f,{}))
    assert not step(0,object_at())['proximity_evidence']
    assert step(.1,object_at(2))['proximity_evidence']
    assert not step(.5,object_at(3))['proximity_evidence']
    assert step(.6,object_at(3))['proximity_evidence']
    changed=object_at(); changed.mask[:]=False; changed.mask[49:52,10:15]=True
    assert not step(.7,changed)['proximity_evidence']
    assert not evidence.apply(None,.8,{})['mask_temporally_stable']


def test_fragmented_or_outside_box_mask_is_suspicion_only():
    lm=landmarks(); f=dict(mouth_visible=True,hand_ok=True,face_width=60)
    for kind in ('outside','fragmented'):
        obj=object_at()
        if kind=='outside': obj.box=(70,70,90,90)
        else:
            obj.mask[:]=False; obj.mask[45:49,10:15]=True; obj.mask[52:56,40:45]=True
            obj.box=(10,45,45,56)
        result=object_proximity(obj,lm,f,{})
        assert not result['mask_geometry_ok'] and not result['proximity_evidence']


def test_pipeline_localization_warms_up_without_dropping_candidate_caution():
    from test_web_runtime import runtime
    from childsafe.evaluation.replay import replay
    import pandas as pd
    demo=runtime('unsafe'); rows=[]
    try:
        for i in range(15):
            _,row,out=demo.pipeline.process(np.full((160,160,3),140,np.uint8),i*.1)
            rows.extend(out['feature_rows'])
            if i==0: assert not row['proximity_evidence'] and out['level']==1
        assert out['level']==2
        assert replay(pd.DataFrame(rows),demo.cfg).iloc[-1].level==out['level']
    finally: demo.close()
