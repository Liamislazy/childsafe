import numpy as np

from childsafe.vision.tracking import PersonTracker
from childsafe.vision.detection import Detection


def person(box):
    return Detection(box, 'person', .99)


def test_distant_replacement_cannot_inherit_id_even_at_high_fps():
    tr = PersonTracker(hold_s=3)
    a, _ = tr.update([person((0,0,100,200))], 0)
    b, retired = tr.update([person((800,0,900,200))], .1)
    assert a[0].person_id != b[0].person_id
    assert a[0].person_id in retired


def test_low_iou_recovery_keeps_display_id_but_discards_risk_history():
    tr = PersonTracker()
    a, _ = tr.update([person((0,0,200,400))], 0)
    b, _ = tr.update([person((130,0,330,400))], .15)
    assert a[0].person_id == b[0].person_id and b[0].history_reset


def test_scale_jump_and_nonmonotonic_time_do_not_reuse_identity():
    for box,t in [((0,0,400,800),.1), ((0,0,100,200),0)]:
        tr = PersonTracker()
        a, _ = tr.update([person((0,0,100,200))], 0)
        b, _ = tr.update([person(box)], t)
        assert a[0].person_id != b[0].person_id


def test_pipeline_discards_geometry_quality_and_alert_on_recovery():
    from test_pipeline import FakeLandmarks
    from childsafe.pipeline import FramePipeline
    from childsafe.config import load_config
    class Det:
        ok = True
        box = (0,0,200,400)
        def detect(self,rgb): return [person(self.box)]
    det = Det()
    pipe = FramePipeline(load_config(),None,detector=det,landmark_factory=FakeLandmarks)
    try:
        rgb = np.zeros((480,640,3),np.uint8)
        _,_,first = pipe.process(rgb,0)
        pid = first['persons'][0]['person_id']
        state = pipe.states[pid]
        state['processor'].last_level = 2
        state['processor'].last_evidence_t = 0
        state['quality'].prev_mouth = np.array([1,2])
        det.box = (130,0,330,400)
        _,_,out = pipe.process(rgb,.15)
        assert out['feature_rows'][0]['history_reset']
        assert out['persons'][0]['level'] == 0
        assert pipe.states[pid]['quality'].prev_mouth is None
    finally: pipe.close()
