"""A5: image corruptions with a 1-5 severity scale.

Used two ways: (1) as random training augmentation (``RobustAug``), (2) at fixed
severity on the test set to draw degradation curves (accuracy vs severity).
Severity 0 means no corruption. The parameter tables are our own choice, loosely
following the ImageNet-C idea; they are not taken from that benchmark.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

SEVERITY = {
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],   # std, image in [0,1]
    "iso_noise":      [0.04, 0.08, 0.12, 0.18, 0.26],   # shot + colour noise
    "motion_blur":    [5, 9, 13, 19, 27],               # kernel length (px at 224)
    "low_light":      [1.5, 2.0, 2.6, 3.3, 4.2],        # gamma > 1 darkens
    "occlusion":      [0.08, 0.15, 0.22, 0.30, 0.40],   # erased area fraction
    "jpeg":           [60, 40, 25, 15, 8],              # quality
}
KINDS = list(SEVERITY)


def _rng(seed):
    return np.random.default_rng(seed)


def corrupt(img: np.ndarray, kind: str, severity: int, seed: int | None = None) -> np.ndarray:
    """img: HxWx3 uint8 RGB. Returns uint8 RGB."""
    if severity <= 0:
        return img
    p = SEVERITY[kind][severity - 1]
    rng = _rng(seed)
    x = img.astype(np.float32) / 255.0
    if kind == "gaussian_noise":
        x = x + rng.normal(0, p, x.shape)
    elif kind == "iso_noise":
        s = 1.0 / p ** 2                      # shot-noise std is p at full brightness
        x = rng.poisson(x * s) / s + rng.normal(0, p / 2, x.shape)
    elif kind == "motion_blur":
        k = int(p)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        angle = rng.uniform(0, 180)
        m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, m, (k, k))
        kernel /= max(kernel.sum(), 1e-6)
        x = cv2.filter2D(x, -1, kernel)
    elif kind == "low_light":
        x = np.power(x, p)
        x = x + rng.normal(0, 0.01 * p, x.shape)
    elif kind == "occlusion":
        h, w = x.shape[:2]
        area = p * h * w
        ar = rng.uniform(0.5, 2.0)
        eh, ew = int(min(h, np.sqrt(area * ar))), int(min(w, np.sqrt(area / ar)))
        y0, x0 = rng.integers(0, h - eh + 1), rng.integers(0, w - ew + 1)
        x[y0:y0 + eh, x0:x0 + ew] = rng.uniform(0, 1, 3)
    elif kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", (np.clip(x, 0, 1) * 255).astype(np.uint8)[..., ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, int(p)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()
    else:
        raise ValueError(kind)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


class FixedCorruption:
    """PIL -> PIL transform with a fixed kind/severity. The seed advances per call, so each image gets
    different noise / occlusion placement, and a fresh instance reproduces the same sequence."""

    def __init__(self, kind: str, severity: int, seed: int = 0):
        self.kind, self.severity, self.seed = kind, severity, seed

    def __call__(self, img: Image.Image) -> Image.Image:
        self.seed += 1
        return Image.fromarray(corrupt(np.asarray(img.convert("RGB")), self.kind, self.severity, self.seed))


class RobustAug:
    """Training-time A5 augmentation: with prob p apply one random corruption at severity 1-3."""

    KINDS = ("iso_noise", "motion_blur", "low_light", "gaussian_noise", "jpeg")

    def __init__(self, p: float = 0.5, max_severity: int = 3):
        self.p, self.max_severity = p, max_severity

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        kind = random.choice(self.KINDS)
        sev = random.randint(1, self.max_severity)
        return Image.fromarray(corrupt(np.asarray(img.convert("RGB")), kind, sev, random.randrange(1 << 30)))
