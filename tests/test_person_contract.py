from types import SimpleNamespace
import numpy as np
from childsafe.vision.people import detect_person


def test_crop_rejects_another_persons_landmarks_and_keeps_owned_hands():
    class Detector:
        def detect(self,rgb,t):
            # Crop origin (100,20), one face outside this person's original box.
            return SimpleNamespace(t=t,face=np.tile([150.,40.,0.],(478,1)),
                                   hands=[np.tile([40.,50.,0.],(21,1)),np.tile([160.,50.,0.],(21,1))],
                                   handedness=['Left','Right'])
    lm=detect_person(Detector(),np.zeros((300,400,3),np.uint8),(100,20,200,220),0,{'zoom':False})
    assert lm.face is None and len(lm.hands)==1 and lm.handedness==['Left']
    np.testing.assert_array_equal(lm.hands[0][0],[140,70,0])


def test_reused_landmarker_uses_stateless_detection_on_each_crop(monkeypatch):
    from childsafe.vision import landmarks as mod
    modes=[]; seen=[]
    class Task:
        def detect(self,image):
            seen.append(float(image.numpy_view().mean()))
            return SimpleNamespace(face_landmarks=[],hand_landmarks=[],handedness=[])
        def detect_for_video(self,*args):
            raise AssertionError('pooled per-person model must not retain VIDEO tracker state')
        def close(self): pass
    def create(options): modes.append(options.running_mode); return Task()
    monkeypatch.setattr(mod.mpv.FaceLandmarker,'create_from_options',create)
    monkeypatch.setattr(mod.mpv.HandLandmarker,'create_from_options',create)
    with mod.LandmarkDetector('face.task','hand.task') as detector:
        detector.detect(np.full((30,30,3),40,np.uint8),1)
        detector.detect(np.full((30,30,3),180,np.uint8),1.1)
    assert modes==[mod.mpv.RunningMode.IMAGE]*2 and seen==[40,40,180,180]
