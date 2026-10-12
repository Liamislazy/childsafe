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
    mask: np.ndarray | None = None  # full-image binary mask (objects only; people use the box)


def _pick_device(device):
    if device not in (None, '', 'auto'):
        return device
    try:
        import torch
        return 0 if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


class SegmentationDetector:
    supports_roi = True   # FramePipeline may run a second, zoomed pass on hands/mouth crops
    def __init__(self, path, confidence=.5, object_confidence=None, device='auto', imgsz=640):
        """confidence: minimum score for people. object_confidence: minimum score for every other
        class (thin/small objects such as forks or knives often score below the person threshold;
        low-score objects only ever reach the CAUTION 'candidate' path, see quality.min_*_mask_conf)."""
        self.model = None
        self.error = ''
        self.checksum = ''
        self.confidence = confidence
        self.object_confidence = confidence if object_confidence is None else object_confidence
        self.device = 'cpu'
        self.imgsz = imgsz
        try:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError('local segmentation weights missing')
            self.checksum = hashlib.sha256(p.read_bytes()).hexdigest()
            from ultralytics import YOLO
            self.model = YOLO(str(p), task='segment')
            self.device = _pick_device(device)
        except Exception as exc:
            self.error = type(exc).__name__  # never log source credentials or exception URLs

    def detect(self, rgb):
        if self.model is None:
            return []
        person_conf = self.confidence
        object_conf = getattr(self, 'object_confidence', person_conf)
        try:
            # Ultralytics numpy inputs are BGR. retina_masks preserves original resolution.
            result = self.model.predict(rgb[:, :, ::-1].copy(), conf=min(person_conf, object_conf),
                                        retina_masks=True, verbose=False,
                                        device=getattr(self, 'device', 'cpu'),
                                        imgsz=getattr(self, 'imgsz', 640))[0]
            detections = []
            for i, box in enumerate(result.boxes):
                label = result.names[int(box.cls.item())]
                score = float(box.conf.item())
                if score < (person_conf if label == 'person' else object_conf):
                    continue
                mask = None
                # Masks are only used for held objects; skip the GPU->CPU copy for people.
                if result.masks is not None and label != 'person':
                    mask = result.masks.data[i].cpu().numpy() > .5
                    if mask.shape != rgb.shape[:2]:
                        mask = None
                detections.append(Detection(tuple(box.xyxy[0].cpu().tolist()), label, score, mask))
            self.error = ''
            return detections
        except Exception as exc:
            self.error = type(exc).__name__
            return []

    @property
    def ok(self):
        return self.model is not None and not self.error
