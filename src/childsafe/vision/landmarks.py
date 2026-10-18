"""MediaPipe Tasks wrappers (Face Landmarker + Hand Landmarker), stateless IMAGE mode.

MediaPipe >= 0.10.2x removed the legacy ``mp.solutions`` API, so this uses the Tasks API,
which needs the .task model files (scripts/download_mediapipe_models.py).
Landmarks are returned in pixel units: x*W, y*H, z*W (MediaPipe's z uses the x scale).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv


@dataclass
class FrameLandmarks:
    t: float
    width: int
    height: int
    face: np.ndarray | None = None                        # (478, 3)
    hands: list[np.ndarray] = field(default_factory=list)  # each (21, 3)
    handedness: list[str] = field(default_factory=list)
    face_matrix: np.ndarray | None = None


class LandmarkDetector:
    def __init__(self, face_model: str, hand_model: str, num_hands: int = 2,
                 min_face_conf: float = 0.5, min_hand_conf: float = 0.5):
        # Pooled instances move across temporary person IDs. VIDEO-mode internal
        # tracking cannot be reset through the public API; IMAGE avoids carrying
        # a previous person's hidden tracker state into another person's crop.
        mode = mpv.RunningMode.IMAGE
        self.face = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model), running_mode=mode, num_faces=1,
            min_face_detection_confidence=min_face_conf, min_face_presence_confidence=min_face_conf,
            output_facial_transformation_matrixes=True))
        self.hand = mpv.HandLandmarker.create_from_options(mpv.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model), running_mode=mode, num_hands=num_hands,
            min_hand_detection_confidence=min_hand_conf, min_hand_presence_confidence=min_hand_conf))

    def detect(self, rgb: np.ndarray, t: float) -> FrameLandmarks:
        """rgb: HxWx3 uint8 RGB frame, t: seconds (monotonic)."""
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        fr = self.face.detect(image)
        hr = self.hand.detect(image)
        scale = np.array([w, h, w], dtype=np.float32)
        out = FrameLandmarks(t=t, width=w, height=h)
        if fr.face_landmarks:
            out.face = np.array([[p.x, p.y, p.z] for p in fr.face_landmarks[0]], np.float32) * scale
            if fr.facial_transformation_matrixes:
                out.face_matrix = np.asarray(fr.facial_transformation_matrixes[0])
        for lm, hd in zip(hr.hand_landmarks, hr.handedness):
            out.hands.append(np.array([[p.x, p.y, p.z] for p in lm], np.float32) * scale)
            out.handedness.append(hd[0].category_name if hd else "")
        return out

    def close(self):
        self.face.close()
        self.hand.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
