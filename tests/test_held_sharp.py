import numpy as np

from childsafe.alerts.processor import AlertProcessor, CAUTION, DANGER
from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection
from test_observation import f
from test_pipeline import FakeLandmarks


def far_from_mouth(t, **kw):
    """Hand and object well away from the mouth: the mouth rule alone gives no alert."""
    row = f(t, **kw)
    row.update(d_face_width=2., object_d_relative=2., proximity_evidence=False, object_present=True)
    return row


def test_confident_knife_in_hand_is_danger_away_from_mouth():
    proc = AlertProcessor(load_config())
    levels = [proc.step(far_from_mouth(i*.1, held_sharp_label='knife', held_sharp_conf=.8,
                                       held_sharp_confident=True))['level'] for i in range(4)]
    assert levels[0] < DANGER and levels[-1] == DANGER
    out = proc.step(far_from_mouth(.4, held_sharp_label='scissors', held_sharp_conf=.8, held_sharp_confident=True))
    assert out['level'] == DANGER and out['reason'] == 'scissors held in hand'


def test_weak_or_ambiguous_or_gappy_detection_is_not_danger():
    cfg = load_config()
    weak, ambiguous, gappy = AlertProcessor(cfg), AlertProcessor(cfg), AlertProcessor(cfg)
    for i in range(10):
        out = weak.step(far_from_mouth(i*.1, held_sharp_label='knife', held_sharp_conf=.45, held_sharp_confident=False))
        assert out['level'] == CAUTION and out['reason'] == 'possible knife in hand'
        out = ambiguous.step(far_from_mouth(i*.1, held_sharp_label='knife', held_sharp_confident=True,
                                            association_ambiguous=True))
        assert out['level'] < DANGER
        # Frames further apart than temporal.max_gap_s never accumulate held time.
        out = gappy.step(far_from_mouth(i*.3, held_sharp_label='knife', held_sharp_confident=True))
        assert out['level'] < DANGER


class VisibleLandmarks(FakeLandmarks):
    def detect(self, rgb, t):
        out = super().detect(rgb, t)
        face = np.zeros((478,3))
        face[234,:2] = [20,40]; face[454,:2] = [100,40]
        face[[78,13,308,14],:2] = [[50,50],[60,45],[70,50],[60,55]]
        face[[33,133],:2] = [40,30]; face[[362,263],:2] = [80,30]
        out.face, out.face_matrix = face, np.eye(4)
        out.hands = [np.column_stack((np.linspace(20,50,21),np.linspace(160,190,21),np.zeros(21)))]
        return out


def run(objects, frames=5):
    class Detector:
        ok = True
        def detect(self, rgb):
            return [Detection((10,20,150,220),'person',.9), *objects]
    cfg = load_config(); cfg['landmarks']['zoom'] = False
    pipe = FramePipeline(cfg, None, detector=Detector(), landmark_factory=VisibleLandmarks,
                         classifier=lambda *args: {})
    rgb = np.random.default_rng(0).integers(40,210,(240,340,3),dtype=np.uint8)
    for i in range(frames):
        _, _, out = pipe.process(rgb, i*.1)
    pipe.close()
    return out['persons'][0]


def test_pipeline_links_knife_in_hand_to_danger_but_not_a_knife_elsewhere():
    held = run([Detection((30,150,70,200),'knife',.8)])
    assert held['level'] == DANGER and held['reason'] == 'knife held in hand'
    on_table = run([Detection((280,200,330,230),'knife',.8)])
    assert on_table['level'] < DANGER
    fork = run([Detection((30,150,70,200),'fork',.8)])
    assert fork['reason'] != 'fork held in hand'
