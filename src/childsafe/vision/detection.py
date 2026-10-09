"""Local-only instance segmentation adapter. Never implicitly downloads weights."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    label: str
    confidence: float
    mask: np.ndarray | None = None  # full-image binary mask


class SegmentationDetector:
    def __init__(self, path, confidence=.5):
        self.model = None
        self.error = ''
        self.checksum = ''
        self.confidence = confidence
        try:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError('local segmentation weights missing')
            self.checksum = hashlib.sha256(p.read_bytes()).hexdigest()
            from ultralytics import YOLO
            self.model = YOLO(str(p), task='segment')
        except Exception as exc:
            self.error = type(exc).__name__  # never log source credentials or exception URLs

    def detect(self, rgb):
        if self.model is None:
            return []
        try:
            # Ultralytics numpy inputs are BGR. retina_masks preserves original resolution.
            result = self.model.predict(rgb[:, :, ::-1].copy(), conf=self.confidence,
                                        retina_masks=True, verbose=False)[0]
            detections = []
            for i, box in enumerate(result.boxes):
                mask = None
                if result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy() > .5
                    if mask.shape != rgb.shape[:2]:
                        mask = None
                detections.append(Detection(tuple(box.xyxy[0].cpu().tolist()),
                                            result.names[int(box.cls.item())],
                                            float(box.conf.item()), mask))
            self.error = ''
            return detections
        except Exception as exc:
            self.error = type(exc).__name__
            return []

    @property
    def ok(self):
        return self.model is not None and not self.error
