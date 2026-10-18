"""Real model timings with a synthetic resolved-object fixture, not a camera accuracy test.

python scripts/benchmark_optimizations.py --ckpt outputs/with_other/runs/efficientnet_b0_group_robust/best.pt
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection, SegmentationDetector
from childsafe.vision.object_classifier import ObjectClassifier


class SyntheticLandmarks:
    def detect(self, rgb, t):
        face = np.zeros((478, 3))
        face[234, :2] = [20, 40]; face[454, :2] = [100, 40]
        face[[78, 13, 308, 14], :2] = [[50, 50], [60, 45], [70, 50], [60, 55]]
        face[[33, 133], :2] = [40, 30]; face[[362, 263], :2] = [80, 30]
        hand = np.column_stack((np.linspace(50, 70, 21), np.linspace(50, 65, 21), np.zeros(21)))
        return SimpleNamespace(t=t, width=rgb.shape[1], height=rgb.shape[0], face=face,
                               hands=[hand], face_matrix=np.eye(4))
    def close(self): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--samples', type=int, default=15)
    ap.add_argument('--cpu-threads', type=int, default=0, help='0 preserves detector library defaults')
    ap.add_argument('--out', default='outputs/optimization_benchmark.json')
    args = ap.parse_args()
    if args.samples < 2: raise ValueError('samples must be >= 2')
    if args.cpu_threads < 0: raise ValueError('cpu-threads must be >= 0')
    cfg = load_config()
    if args.cpu_threads: torch.set_num_threads(args.cpu_threads)
    detector = SegmentationDetector(cfg['detector']['weights'], cfg['detector']['min_conf'], device='cpu', cpu_threads=args.cpu_threads)
    if not detector.ok: raise RuntimeError(detector.error)
    classifier = ObjectClassifier(args.ckpt, device='cpu')
    rgb = np.full((480, 640, 3), 140, np.uint8)
    rgb[::4, ::4] = 100
    class TimedDetector:
        ok, supports_roi = True, True
        def __init__(self): self.calls = 0
        def detect(self, image):
            self.calls += 1
            detector.detect(image)  # actual YOLO inference, not a timer/sleep approximation
            if not detector.ok: raise RuntimeError(detector.error)
            if image.shape != rgb.shape: return []
            mask = np.zeros(image.shape[:2], bool); mask[45:60, 65:80] = True
            return [Detection((0, 0, 640, 480), 'person', .99), Detection((65, 45, 80, 60), 'knife', .99, mask)]
    runs = {}
    for skip in (False, True):
        config = copy.deepcopy(cfg); config['quality']['min_sharpness'] = 0
        config['detector']['roi_skip_resolved'] = skip
        measured = TimedDetector()
        pipe = FramePipeline(config, None, detector=measured, landmark_factory=SyntheticLandmarks, classifier=classifier)
        times = []
        try:
            for i in range(args.samples+3):
                start = time.perf_counter(); pipe.process(rgb, i*.1); elapsed = time.perf_counter()-start
                if i >= 3: times.append(elapsed)
        finally: pipe.close()
        runs['optimized' if skip else 'previous_roi_policy'] = {
            'median_ms': float(np.median(times)*1000), 'p90_ms': float(np.percentile(times, 90)*1000),
            'detector_calls_per_frame': measured.calls/(args.samples+3), 'samples': args.samples}
    report = dict(input_scope='Synthetic person/object/mouth fixtures; real YOLO and classifier inference on CPU',
                  cpu_threads=torch.get_num_threads(), torch_version=torch.__version__,
                  detector_sha256=detector.checksum,
                  classifier_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(), timings=runs,
                  limitations=['No real webcam accuracy, no multi-person recall, no camera calibration.',
                               'Resolved-object workload only; missed/weak masks still perform ROI inference.',
                               'Landmarks are synthetic: timings exclude MediaPipe landmark detection.'])
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8'); print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
