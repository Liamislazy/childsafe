"""Frame pipeline shared by the live webcam app and offline feature extraction.

frame -> landmarks (Module B) -> geometry -> [cascade gate] -> hand-crop classifier (Module A)
      -> AlertProcessor (smoothing, temporal rule, alert matrix)
"""
from __future__ import annotations

import math

import numpy as np

from childsafe.alerts.processor import AlertProcessor, Components
from childsafe.vision.geometry import GeometryExtractor
from childsafe.vision.landmarks import LandmarkDetector
from childsafe.vision.object_classifier import ObjectClassifier, hand_crop_box

OBJ_COLUMNS = ["p_unsafe_raw", "p_unsafe_cal", "obj_class", "obj_conf"]


class FramePipeline:
    def __init__(self, cfg: dict, ckpt: str | None, components: Components | None = None,
                 classify_all: bool = False, classify_input: str = "hand_crop"):
        lc = cfg["landmarks"]
        self.cfg = cfg
        self.detector = LandmarkDetector(cfg["paths"]["face_model"], cfg["paths"]["hand_model"],
                                         lc["num_hands"], lc["min_face_conf"], lc["min_hand_conf"])
        self.geometry = GeometryExtractor()
        self.classifier = ObjectClassifier(ckpt) if ckpt else None
        self.processor = AlertProcessor(cfg, components)
        self.classify_all = classify_all      # offline: classify every hand frame so replay can ablate D2
        self.classify_input = classify_input

    def process(self, rgb: np.ndarray, t: float):
        lm = self.detector.detect(rgb, t)
        f = self.geometry(lm)
        obj = None
        want = self.processor.wants_classification(f)
        if self.classifier is not None and (want or (self.classify_all and f["hand_ok"])):
            box = hand_crop_box(f, lm.width, lm.height, self.cfg["cascade"]["crop_scale"]) \
                if self.classify_input == "hand_crop" else None
            if box is not None or self.classify_input == "frame":
                obj = self.classifier(rgb, box)
        out = self.processor.step(f, obj if want else None)
        row = {**f, **{k: (obj[k] if obj else (math.nan if k != "obj_class" else "")) for k in OBJ_COLUMNS}}
        return lm, row, out

    def close(self):
        self.detector.close()
