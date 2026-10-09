"""Module A at run time: classify a square crop around the hand (or the whole frame)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from childsafe.classifier.calibration import softmax
from childsafe.classifier.models import load_checkpoint
from childsafe.data.dataset import eval_transform


def hand_crop_box(f: dict, width: int, height: int, scale: float):
    """Square box around the hand bbox, `scale` times its larger side, clipped to the frame."""
    if np.isnan(f.get("hand_x0", np.nan)):
        return None
    cx, cy = (f["hand_x0"] + f["hand_x1"]) / 2, (f["hand_y0"] + f["hand_y1"]) / 2
    side = max(f["hand_x1"] - f["hand_x0"], f["hand_y1"] - f["hand_y0"]) * scale
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
    return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None


class ObjectClassifier:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ck = load_checkpoint(ckpt_path, self.device)
        self.classes, self.temperature = ck["classes"], ck["temperature"]
        self.unsafe = np.array([ck["binary_map"][c] == "unsafe" for c in self.classes])
        self.unknown = np.array([ck["binary_map"][c] == "unknown" or c in ("other", "unknown", "empty_hand")
                                 for c in self.classes])
        self.tf = eval_transform(ck["img_size"], preprocessing=ck.get("preprocessing", "center_crop"))

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, box=None) -> dict:
        crop = rgb if box is None else rgb[box[1]:box[3], box[0]:box[2]]
        x = self.tf(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        logits = self.model(x).float().cpu().numpy()
        raw, cal = softmax(logits, 1.0)[0], softmax(logits, self.temperature)[0]
        k = int(cal.argmax())
        top = np.argsort(-cal)[:3]
        return {"p_unsafe_raw": float(raw[self.unsafe].sum()), "p_unsafe_cal": float(cal[self.unsafe].sum()),
                "obj_class": self.classes[k], "obj_conf": float(cal[k]),
                "p_unknown_cal": float(cal[self.unknown].sum()),
                "top3": [(self.classes[i], float(cal[i])) for i in top]}
