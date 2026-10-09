from types import SimpleNamespace
import math
import numpy as np

from childsafe.alerts.processor import AlertProcessor, DANGER
from childsafe.config import load_config
from childsafe.vision.detection import Detection
from childsafe.vision.elevated import mask_region_distance, object_proximity, ElevatedQuality


def landmarks():
    face = np.zeros((478,3))
    face[[78,13,308,14],:2] = [[45,50],[50,45],[55,50],[50,55]]
    return SimpleNamespace(face=face,face_matrix=np.eye(4),hands=[np.array([[10,10,0],[30,30,0]])],
                           width=100,height=100,t=0)


def test_long_object_near_mouth_with_hand_far_uses_mask():
    mask = np.zeros((100,100),bool)
    mask[49:52,10:50] = True
    lm = landmarks()
    f = dict(mouth_visible=True,hand_ok=True,face_width=60)
    out = object_proximity(Detection((10,49,50,52),'knife',.9,mask),lm,f,{})
    assert out['proximity_evidence'] and out['object_d_relative'] == 0
    assert np.linalg.norm(lm.hands[0][0,:2]-[50,50]) > 50


def test_no_mask_or_no_visible_mouth_has_no_proximity():
    lm = landmarks()
    f = dict(mouth_visible=True,hand_ok=True,face_width=60)
    obj = Detection((0,0,80,80),'unknown',.9)
    assert not object_proximity(obj,lm,f,{})['proximity_evidence']
    obj.mask = np.ones((100,100),bool)
    f['mouth_visible'] = False
    assert math.isnan(object_proximity(obj,lm,f,{})['object_d_relative'])


def test_bounding_box_or_hand_near_head_cannot_confirm_danger():
    p = AlertProcessor(load_config())
    for i in range(60):
        f = dict(t=i/30,schema_version=2,face_ok=True,mouth_visible=True,hand_ok=True,
                 detector_ok=True,classifier_ok=True,d_face_width=.1,proximity_evidence=False)
        out = p.step(f,dict(p_unsafe_cal=.9,p_unsafe_raw=.9,obj_conf=.9,obj_class='knife'))
        assert out['level'] != DANGER


def test_back_view_small_face_and_blur_are_unobservable():
    lm = landmarks()
    lm.face_matrix[:3,:3] = np.diag([1,-1,-1])
    rng = np.random.default_rng(0)
    rgb = rng.integers(40,210,(100,100,3),dtype=np.uint8)
    q = ElevatedQuality({})
    assert not q.measure(rgb,lm,dict(face_width=60))['mouth_visible']
    lm.face_matrix = np.eye(4)
    assert not q.measure(rgb,lm,dict(face_width=10))['mouth_visible']
    assert not q.measure(np.full_like(rgb,120),lm,dict(face_width=60))['mouth_visible']


def test_region_distance_is_not_box_center_distance():
    mask = np.zeros((100,100),bool); mask[50,80] = True
    assert abs(mask_region_distance(mask, [[45,45],[55,45],[55,55],[45,55]])-25) < 1e-8


def test_uncertain_mask_tip_requests_attention_without_confirmed_dwell():
    lm = landmarks()
    mask = np.zeros((100,100),bool); mask[49:52,10:50] = True
    base = dict(schema_version=2,face_ok=True,mouth_visible=True,hand_ok=True,face_width=60,
                detector_ok=True,classifier_ok=True,object_present=True,d_face_width=.7)
    proximity = object_proximity(Detection((10,49,50,52),'toothbrush',.55,mask),lm,base,{})
    assert not proximity['mask_reliable'] and not proximity['proximity_evidence']
    assert proximity['candidate_object_d_relative'] == 0
    processor = AlertProcessor(load_config())
    for i in range(20):
        out = processor.step(dict(base,**proximity,t=i*.1))
        assert out['level'] == 1 and out['dwell'] == 0 and not out['notify']
    for overrides in [dict(mouth_visible=False),dict(association_ambiguous=True),
                      dict(hand_ok=False),dict(source_ok=False),dict(object_present=False)]:
        processor.reset()
        assert processor.step(dict(base,**proximity,t=3,**overrides))['level'] == 0
