"""Recover distant localization without reintroducing bare-hand alerts or fake masks."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from childsafe.alerts.processor import AlertProcessor
from childsafe.config import load_config
from childsafe.pipeline import FramePipeline,SceneLandmarks
from childsafe.vision.detection import Detection
from childsafe.vision.elevated import object_proximity,box_region_distance
from childsafe.vision.people import detect_person
from childsafe.vision.tracking import Track
from test_elevated import landmarks


@pytest.mark.parametrize('kind',['empty','sparse','wrong_shape','fragmented'])
def test_detected_box_survives_an_unusable_far_mask_as_caution_only(kind):
    lm=landmarks(); mask=np.zeros((100,100),bool)
    if kind=='sparse': mask[50,50]=True
    if kind=='wrong_shape': mask=np.ones((2,2),bool)
    if kind=='fragmented': mask[43:46,43:46]=True; mask[54:57,54:57]=True
    before=mask.copy(); obj=Detection((40,40,60,60),'knife',.95,mask)
    row=dict(schema_version=2,face_ok=True,hand_ok=True,mouth_visible=True,face_width=60,
             detector_ok=True,classifier_ok=True,object_present=True,d_face_width=.8)
    proximity=object_proximity(obj,lm,row,{})
    assert proximity['candidate_localization']=='box'
    assert proximity['candidate_object_d_relative']==0 and not proximity['proximity_evidence']
    assert not proximity['mask_reliable'] and math.isnan(proximity['object_d_relative'])
    assert np.array_equal(mask,before) and obj.mask is mask
    proc=AlertProcessor(load_config()); states=[]
    for i in range(15): states.append(proc.step(dict(row,**proximity,t=i*.1))['level'])
    assert states[:2]==[0,0] and states[-1]==1 and 2 not in states


def test_box_distance_uses_region_intersection_not_box_center():
    mouth=[[45,50],[50,45],[55,50],[50,55]]
    assert box_region_distance((0,49,100,51),mouth,100,100)==0
    assert box_region_distance((80,45,90,55),mouth,100,100)==pytest.approx(25)
    assert math.isnan(box_region_distance((110,0,120,10),mouth,100,100))


def face_at(cx,cy,fw=60):
    face=np.zeros((478,3)); face[:,:2]=[cx,cy]
    face[234,:2]=[cx-fw/2,cy]; face[454,:2]=[cx+fw/2,cy]
    face[[13,14,78,308],:2]=[[cx,cy+8],[cx,cy+12],[cx-5,cy+10],[cx+5,cy+10]]
    return face


class HandRetry:
    def __init__(self,fw=60,fail=False): self.calls=[];self.fw=fw;self.fail=fail
    def detect(self,rgb,t):
        h,w=rgb.shape[:2]
        return SimpleNamespace(t=t,width=w,height=h,face=face_at(w/2,h/3,self.fw),
                               hands=[],handedness=[],face_matrix=np.eye(4))
    def detect_hands(self,rgb,t):
        self.calls.append(rgb.shape[:2])
        if self.fail: raise RuntimeError('must not expose private details')
        h,w=rgb.shape[:2]
        hand=np.column_stack((np.linspace(w/2-8,w/2+8,21),np.linspace(h/2,h/2+16,21),np.zeros(21)))
        return SimpleNamespace(t=t,face=None,hands=[hand],handedness=['Right'])
    def close(self): pass


def test_hand_only_roi_retry_maps_back_to_original_pixels():
    rgb=np.full((480,640,3),140,np.uint8); detector=HandRetry()
    cfg=dict(zoom=False,hand_roi_retry=True)
    lm=detect_person(detector,rgb,(0,0,640,480),0,cfg,lone=True)
    assert detector.calls==[(180,180)] and lm.hand_roi_found==1 and lm.hand_roi_attempted
    x0,y0,_,_=lm.hand_roi_box
    assert np.allclose(lm.hands[0][0,:2],[x0+82,y0+90])
    assert lm.width==640 and lm.height==480


def test_retry_skips_tiny_faces_and_insufficient_deadline_and_preserves_primary_on_error(monkeypatch):
    rgb=np.full((480,640,3),140,np.uint8); cfg=dict(zoom=False,hand_roi_retry=True)
    tiny=HandRetry(fw=28)
    assert not detect_person(tiny,rgb,(0,0,640,480),0,cfg,lone=True).hand_roi_attempted
    detector=HandRetry()
    monkeypatch.setattr('childsafe.vision.people.time',SimpleNamespace(perf_counter=lambda:1.))
    assert not detect_person(detector,rgb,(0,0,640,480),0,cfg,True,deadline=1.01).hand_roi_attempted
    broken=HandRetry(fail=True)
    result=detect_person(broken,rgb,(0,0,640,480),0,cfg,lone=True)
    assert result.face is not None and result.hand_roi_error=='RuntimeError' and not result.hands


def test_one_hand_retry_per_frame_rotates_across_four_people():
    class Detector:
        ok=True
        def detect(self,rgb): return [Detection((i*200+50,0,i*200+150,300),'person',.99) for i in range(4)]
    created=[]
    def factory():
        item=HandRetry(fw=40);created.append(item);return item
    cfg=load_config();cfg['quality']['min_sharpness']=0
    pipe=FramePipeline(cfg,None,detector=Detector(),landmark_factory=factory)
    rgb=np.full((400,800,3),140,np.uint8)
    try:
        for i in range(4):
            out=pipe.process(rgb,i*.1)[2]
            assert sum(row['hand_roi_attempted'] for row in out['feature_rows'])==1
            assert out['level']==0  # recovered hands alone are not objects
        assert [len(item.calls) for item in created]==[1,1,1,1]
    finally: pipe.close()


def test_separated_hands_no_longer_disable_the_near_mouth_object_search():
    class Detector:
        ok=True;supports_roi=True
        def __init__(self):self.calls=[]
        def detect(self,rgb):self.calls.append(rgb.shape[:2]);return []
    face=face_at(200,190,60)
    def hand(x,y):return np.column_stack((np.linspace(x,x+10,21),np.linspace(y,y+10,21),np.zeros(21)))
    scene=SceneLandmarks(people={1:SimpleNamespace(face=face,hands=[hand(190,195),hand(700,500)])})
    tracks=[Track(1,(0,0,800,600),0)];rgb=np.full((600,800,3),140,np.uint8)
    cfg=load_config();det=Detector();pipe=FramePipeline(cfg,None,detector=det)
    pipe.roi_passes=0;pipe._zoom_detect(rgb,scene,tracks,set())
    assert det.calls==[(192,192)]
    cfg['detector']['roi_per_hand']=False;det.calls.clear();pipe.roi_passes=0
    pipe._zoom_detect(rgb,scene,tracks,set())
    assert not det.calls  # former union of both hands exceeded the crop-size budget


def test_optional_object_search_is_skipped_when_it_cannot_fit_deadline(monkeypatch):
    class Detector:
        ok=True;supports_roi=True
        def detect(self,rgb):raise AssertionError('should preserve frame instead of starting work that cannot fit')
    cfg=load_config();pipe=FramePipeline(cfg,None,detector=Detector())
    hand=np.column_stack((np.linspace(190,200,21),np.linspace(190,200,21),np.zeros(21)))
    scene=SceneLandmarks(people={1:SimpleNamespace(face=face_at(200,190),hands=[hand])})
    monkeypatch.setattr('childsafe.pipeline.time',SimpleNamespace(perf_counter=lambda:1.))
    pipe.roi_passes=0
    assert pipe._zoom_detect(np.zeros((480,640,3),np.uint8),scene,[Track(1,(0,0,640,480),0)],set(),deadline=1.03)==0
    assert pipe.roi_passes==0 and pipe.roi_budget_skips==1
