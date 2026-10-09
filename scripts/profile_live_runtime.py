"""Profile the live-camera runtime on synthetic, non-human frames (no images of people).

    python scripts/profile_live_runtime.py outputs/with_other/runs/efficientnet_b0_group_robust/best.pt

Measures each component and the full FramePipeline with a fake detector returning 1 or 2
(overlapping) people and a held object with a full-resolution mask. Not an accuracy test.
"""
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection, SegmentationDetector
from childsafe.vision.elevated import ElevatedQuality, mask_region_distance
from childsafe.vision.landmarks import LandmarkDetector
from childsafe.vision.object_classifier import ObjectClassifier

CKPT = sys.argv[1]
W, H = 640, 480
cfg = load_config()


def frame():
    img = Image.new('RGB', (W, H), (170, 165, 160))
    d = ImageDraw.Draw(img)
    d.ellipse((260, 80, 380, 220), fill=(210, 170, 140))
    d.rectangle((300, 200, 330, 330), fill=(40, 60, 150))
    return np.asarray(img)


def timeit(fn, n=15, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
    return round(float(np.median(ts)) * 1000, 1)


rgb = frame()
res = {}
s = time.perf_counter()
lmd = LandmarkDetector(cfg['paths']['face_model'], cfg['paths']['hand_model'])
res['mediapipe_init_ms'] = round((time.perf_counter() - s) * 1000, 1)
k = [0]
def lm_call():
    k[0] += 1; lmd.detect(rgb, k[0] / 30)
res['mediapipe_detect_ms'] = timeit(lm_call)

det = SegmentationDetector(cfg['detector']['weights'], cfg['detector']['min_conf'])
assert det.ok, det.error
res['yolo_detect_ms'] = timeit(lambda: det.detect(rgb))

clf = ObjectClassifier(CKPT)
res['classifier_ms'] = timeit(lambda: clf(rgb, (280, 180, 360, 340)))

mask = np.zeros((H, W), bool); mask[150:330, 290:340] = True   # ~9000 px object mask
poly = np.array([[300, 180], [320, 175], [340, 180], [320, 190]], float)
res['mask_distance_ms'] = timeit(lambda: mask_region_distance(mask, poly))
q = ElevatedQuality(cfg['quality'])
res['quality_measure_ms_640x480'] = timeit(lambda: q.measure(rgb, None, {}))
big = np.ascontiguousarray(np.resize(rgb, (720, 1280, 3)))
res['quality_measure_ms_1280x720'] = timeit(lambda: q.measure(big, None, {}))


class FakeDetector:
    """Two overlapping people (tracker marks them ambiguous) or one person, plus a held object."""
    ok, error = True, ''

    def __init__(self, people):
        self.people = people

    def detect(self, rgb):
        boxes = [(200, 40, 440, 470), (330, 40, 600, 470)][:self.people]
        out = [Detection(b, 'person', .9, None) for b in boxes]
        out.append(Detection((290, 150, 340, 330), 'knife', .7, mask.copy()))
        return out


for people in (1, 2):
    pipe = FramePipeline(cfg, None, detector=FakeDetector(people), classifier=clf)
    ts, t = [], 0.
    for i in range(25):
        t += .1
        s = time.perf_counter(); pipe.process(rgb, t); ts.append(time.perf_counter() - s)
    res[f'pipeline_{people}person_median_ms'] = round(float(np.median(ts[5:])) * 1000, 1)
    res[f'pipeline_{people}person_states_alive'] = len(pipe.states)
    res[f'pipeline_{people}person_landmark_instances'] = sum(st['landmarks'] is not None for st in pipe.states.values())
    pipe.close()
lmd.close()
print(json.dumps(res, indent=1))
