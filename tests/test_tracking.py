from types import SimpleNamespace

import numpy as np

from childsafe.vision.detection import Detection
from childsafe.vision.tracking import PersonTracker, to_original, associate_object


def person(box):
    return Detection(box, 'person', .9)


def test_ids_stable_then_crossing_resets_history():
    tr = PersonTracker()
    a, _ = tr.update([person((0,0,100,200)),person((200,0,300,200))], 0)
    b, _ = tr.update([person((2,0,102,200)),person((198,0,298,200))], .1)
    assert [p.person_id for p in a] == [p.person_id for p in b]
    c, retired = tr.update([person((70,0,170,200)),person((90,0,190,200))], .2)
    assert all(p.ambiguous for p in c)
    assert all(p.person_id not in {x.person_id for x in b} for p in c)


def test_crop_coordinates_translate_xy_only():
    lm = SimpleNamespace(face=np.array([[1.,2.,3.]]),hands=[np.array([[4.,5.,6.]])])
    out = to_original(lm, (100,200,300,400), 640, 480)
    np.testing.assert_array_equal(out.face, [[101,202,3]])
    np.testing.assert_array_equal(out.hands[0], [[104,205,6]])
    assert (out.width,out.height) == (640,480)


def test_object_between_two_hands_has_no_owner():
    tr = PersonTracker()
    people, _ = tr.update([person((0,0,100,100)),person((150,0,250,100))], 0)
    landmarks = {p.person_id:SimpleNamespace(hands=[np.array([[120.,50.,0.]])]) for p in people}
    assert associate_object(Detection((110,40,130,60),'knife',.9), people, landmarks) == (None, True)


def test_capacity_and_missing_model():
    tr = PersonTracker()
    people, _ = tr.update([person((i*100,0,i*100+50,100)) for i in range(5)], 0)
    assert len(people) == 4 and all(p.ambiguous for p in people)
    from childsafe.vision.detection import SegmentationDetector
    assert not SegmentationDetector('missing.pt').ok


def test_slow_identity_hold_does_not_recreate_models_or_display_retired_people():
    from childsafe.config import load_config
    from childsafe.pipeline import FramePipeline
    calls = []
    class Detector:
        ok = True
        def detect(self, rgb): return [person((10,10,90,90))]
    class Landmarks:
        def detect(self, rgb, t):
            calls.append(rgb.shape)
            return SimpleNamespace(t=t,face=None,hands=[],width=100,height=100)
        def close(self): pass
    made = []
    def factory():
        made.append(1)
        return Landmarks()
    pipeline = FramePipeline(load_config(),None,detector=Detector(),landmark_factory=factory,
                             classifier=lambda *args:None)
    rgb = np.full((100,100,3),120,np.uint8)
    try:
        ids = []
        for t in [0,1.3,2.6,7]:
            _,_,out = pipeline.process(rgb,t)
            assert len(out['persons']) == 1
            ids.append(out['persons'][0]['person_id'])
        assert ids[0] == ids[1] == ids[2] and ids[3] != ids[2]
        # MediaPipe instances are pooled: the new ID after expiry reuses the released one.
        assert len(made) == 1 and calls == [(100,100,3)]*4
    finally: pipeline.close()
