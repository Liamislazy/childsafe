from types import SimpleNamespace
import numpy as np

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection


class FakeDetector:
    ok = True
    calls = 0
    def detect(self, rgb):
        self.calls += 1
        return [Detection((10,20,150,220),'person',.9),
                Detection((180,20,320,220),'person',.9)]


class FakeLandmarks:
    def detect(self, rgb, t):
        return SimpleNamespace(t=t, width=rgb.shape[1],height=rgb.shape[0],face=None,
                               hands=[],face_matrix=None)
    def close(self):
        pass


def test_per_person_pipeline_and_stale_frame():
    detector = FakeDetector()
    pipe = FramePipeline(load_config(), None, detector=detector, landmark_factory=FakeLandmarks)
    rgb = np.zeros((240,340,3),np.uint8)
    scene, row, out = pipe.process(rgb,0)
    assert len(out['persons']) == 2
    assert len({p['person_id'] for p in out['persons']}) == 2
    assert not out['green']
    pipe.process(rgb,.1,frame_age_s=.5)
    assert detector.calls == 1
    pipe.reset(7)
    _, _, out = pipe.process(rgb,.2)
    assert all(p['session_id'] == 7 for p in out['persons'])
    assert min(p['person_id'] for p in out['persons']) > 2
    pipe.close()


def test_calibrated_complete_detector_can_show_normal_observable_when_hands_are_clear():
    class VisibleLandmarks(FakeLandmarks):
        def detect(self,rgb,t):
            out = super().detect(rgb,t)
            face = np.zeros((478,3))
            face[234,:2] = [20,40]; face[454,:2] = [100,40]
            face[[78,13,308,14],:2] = [[50,50],[60,45],[70,50],[60,55]]
            face[[33,133],:2] = [40,30]; face[[362,263],:2] = [80,30]
            out.face,out.face_matrix = face,np.eye(4)
            out.hands = [np.column_stack((np.linspace(20,50,21),np.linspace(160,190,21),np.zeros(21)))]
            return out
    cfg = load_config(); cfg['quality']['calibrated'] = True
    cfg['detector']['complete_object_coverage'] = True
    pipe = FramePipeline(cfg,None,detector=FakeDetector(),landmark_factory=VisibleLandmarks,
                         classifier=lambda *args: {})
    rgb = np.random.default_rng(0).integers(40,210,(240,340,3),dtype=np.uint8)
    _,_,out = pipe.process(rgb,0)
    assert out['green'] and all(person['observation'] == 'observable' for person in out['persons'])
    pipe.close()
