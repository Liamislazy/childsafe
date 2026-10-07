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
