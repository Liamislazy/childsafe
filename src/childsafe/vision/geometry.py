"""Per-frame geometry: mouth point, scale normalizers (B1) and normalized hand-to-mouth distance.

All distances are 2D image-plane distances divided by a face or hand scale, so they are
RELATIVE, not metric. A hand in front of the face but far from the mouth in depth can
still give a small value (handoff risk 1). ``hand_face_ratio`` is logged as a weak depth cue.
"""
from __future__ import annotations

import numpy as np

from childsafe.vision.landmarks import FrameLandmarks

# Face Landmarker (478-point mesh) indices.
MOUTH = [13, 14, 78, 308]          # inner upper lip, inner lower lip, inner corners
FACE_LEFT, FACE_RIGHT = 234, 454   # cheek extremes -> face width
LEFT_EYE = [33, 133]               # outer / inner corner
RIGHT_EYE = [362, 263]
# Hand Landmarker (21 points)
WRIST, MIDDLE_MCP = 0, 9

NORMALIZERS = ("face_width", "inter_eye", "hand_size")
FEATURE_COLUMNS = ["t", "face_ok", "hand_ok", "n_hands", "mouth_x", "mouth_y", "face_width", "inter_eye",
                   "hand_size", "d_px", "d_face_width", "d_inter_eye", "d_hand_size", "hand_face_ratio",
                   "d_last_mouth", "hand_x0", "hand_y0", "hand_x1", "hand_y1"]


def _dist2d(a, b):
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def face_measures(face: np.ndarray) -> dict:
    mouth = face[MOUTH, :2].mean(0)
    left_eye, right_eye = face[LEFT_EYE, :2].mean(0), face[RIGHT_EYE, :2].mean(0)
    return {"mouth": mouth, "face_width": _dist2d(face[FACE_LEFT], face[FACE_RIGHT]),
            "inter_eye": _dist2d(left_eye, right_eye)}


def hand_to_point(hand: np.ndarray, point: np.ndarray) -> float:
    """Minimum 2D distance from any of the 21 hand landmarks to `point`.
    (Any hand point, not only fingertips, because a held object may be anywhere around the hand.)"""
    return float(np.linalg.norm(hand[:, :2] - point[None, :2], axis=1).min())


class GeometryExtractor:
    """Stateful: remembers the last mouth position so that when the face is lost (often because
    the hand/object covers it) we can still measure hand distance to where the mouth was (B2)."""

    def __init__(self):
        self.last_mouth = None
        self.last_face_width = None

    def __call__(self, lm: FrameLandmarks) -> dict:
        nan = float("nan")
        f = {c: nan for c in FEATURE_COLUMNS}
        f.update(t=lm.t, face_ok=int(lm.face is not None), hand_ok=int(bool(lm.hands)), n_hands=len(lm.hands))
        if lm.face is not None:
            fm = face_measures(lm.face)
            f.update(mouth_x=float(fm["mouth"][0]), mouth_y=float(fm["mouth"][1]),
                     face_width=fm["face_width"], inter_eye=fm["inter_eye"])
            self.last_mouth, self.last_face_width = fm["mouth"], fm["face_width"]
        if lm.hands:
            if lm.face is not None:
                dists = [hand_to_point(h, fm["mouth"]) for h in lm.hands]
                k = int(np.argmin(dists))
                hand = lm.hands[k]
                hs = _dist2d(hand[WRIST], hand[MIDDLE_MCP])
                f.update(d_px=dists[k], hand_size=hs,
                         d_face_width=dists[k] / max(fm["face_width"], 1e-6),
                         d_inter_eye=dists[k] / max(fm["inter_eye"], 1e-6),
                         d_hand_size=dists[k] / max(hs, 1e-6),
                         hand_face_ratio=hs / max(fm["face_width"], 1e-6))
            elif self.last_mouth is not None:
                dists = [hand_to_point(h, self.last_mouth) for h in lm.hands]
                k = int(np.argmin(dists))
                hand = lm.hands[k]
                f["d_last_mouth"] = dists[k] / max(self.last_face_width, 1e-6)
            else:
                hand = lm.hands[0]
            x0, y0 = hand[:, :2].min(0)
            x1, y1 = hand[:, :2].max(0)
            f.update(hand_x0=float(x0), hand_y0=float(y0), hand_x1=float(x1), hand_y1=float(y1))
        return f


def normalized_distance(f: dict, normalizer: str) -> float:
    return f[f"d_{normalizer}"]
