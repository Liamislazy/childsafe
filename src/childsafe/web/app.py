"""Local web demo: compare two Module A checkpoints (e.g. standard vs A5-robust) on the same
webcam frames, each with its own alert processor, plus a photo-upload tester.

Privacy: the server listens on 127.0.0.1 only, frames are processed in memory and never
written to disk. Run with scripts/web_demo.py.
"""
from __future__ import annotations

import base64
import io
import math
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from childsafe.alerts.processor import LEVEL_NAMES, AlertProcessor
from childsafe.data.corruptions import KINDS, corrupt
from childsafe.vision.geometry import MOUTH, GeometryExtractor
from childsafe.vision.landmarks import LandmarkDetector
from childsafe.vision.object_classifier import ObjectClassifier, hand_crop_box

STATIC = Path(__file__).parent / "static"


def _nan_to_none(x):
    return None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x


def _thumb(rgb: np.ndarray, size: int = 160) -> str:
    im = Image.fromarray(rgb)
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _decode(body: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))


class Demo:
    def __init__(self, cfg: dict, models: dict[str, str]):
        self.cfg = cfg
        self.lock = threading.Lock()
        lc = cfg["landmarks"]
        self.detector = LandmarkDetector(cfg["paths"]["face_model"], cfg["paths"]["hand_model"],
                                         lc["num_hands"], lc["min_face_conf"], lc["min_hand_conf"])
        self.classifiers = {name: ObjectClassifier(path) for name, path in models.items()}
        self.paths = models
        self.t0 = time.perf_counter()
        self.reset()

    def reset(self):
        self.geometry = GeometryExtractor()
        self.processors = {name: AlertProcessor(self.cfg) for name in self.classifiers}

    def _crop(self, rgb, f, mode):
        h, w = rgb.shape[:2]
        if mode == "full":
            return None
        if mode == "center":
            s = int(min(h, w) * 0.6)
            return ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        return hand_crop_box(f, w, h, self.cfg["cascade"]["crop_scale"])

    def frame(self, body: bytes, kind: str, severity: int, mode: str) -> dict:
        rgb = _decode(body)
        if kind in KINDS and severity > 0:
            rgb = corrupt(rgb, kind, severity, seed=int(time.time() * 1000) % (1 << 30))
        with self.lock:
            t = time.perf_counter() - self.t0
            t_start = time.perf_counter()
            lm = self.detector.detect(rgb, t)
            f = self.geometry(lm)
            box = self._crop(rgb, f, mode)
            classify = mode != "hand" or box is not None
            crop = rgb if box is None else rgb[box[1]:box[3], box[0]:box[2]]
            out = {"width": rgb.shape[1], "height": rgb.shape[0], "box": box,
                   "face": lm.face is not None, "hands": [h[:, :2].round(1).tolist() for h in lm.hands],
                   "mouth": lm.face[MOUTH, :2].mean(0).round(1).tolist() if lm.face is not None else None,
                   "crop": _thumb(crop) if classify else None, "models": {}}
            for name, clf in self.classifiers.items():
                obj = clf(rgb, box) if classify else None
                res = self.processors[name].step(f, obj)
                out["models"][name] = {
                    "level": res["level"], "level_name": LEVEL_NAMES[res["level"]], "reason": res["reason"],
                    "d": _nan_to_none(res["d"]), "near": bool(res["near"]), "dwell": res["dwell"],
                    "lost_near": bool(res["lost_near"]), "obj_status": res["obj_status"],
                    "p_unsafe": obj["p_unsafe_cal"] if obj else None,
                    "top3": obj["top3"] if obj else [],
                }
            out["ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return out

    def image(self, body: bytes, kind: str, severity: int) -> dict:
        rgb = _decode(body)
        if kind in KINDS and severity > 0:
            rgb = corrupt(rgb, kind, severity, seed=0)
        with self.lock:
            res = {name: clf(rgb) for name, clf in self.classifiers.items()}
        return {"preview": _thumb(rgb, 320),
                "models": {n: {"p_unsafe": r["p_unsafe_cal"], "top3": r["top3"],
                               "status": "uncertain" if max(r["p_unsafe_cal"], 1 - r["p_unsafe_cal"])
                               < self.cfg["calibration"]["reject_threshold"]
                               else ("unsafe" if r["p_unsafe_cal"] >= 0.5 else "safe")}
                           for n, r in res.items()}}


def create_app(demo: Demo) -> FastAPI:
    app = FastAPI(title="childsafe demo")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {"models": {n: Path(p).parent.name for n, p in demo.paths.items()}, "corruptions": KINDS,
                "thresholds": {k: demo.cfg["temporal"][k] for k in ("d_enter", "d_exit")}}

    @app.post("/api/frame")
    async def frame(request: Request, kind: str = "", severity: int = 0, mode: str = "hand"):
        return JSONResponse(demo.frame(await request.body(), kind, severity, mode))

    @app.post("/api/image")
    async def image(request: Request, kind: str = "", severity: int = 0):
        return JSONResponse(demo.image(await request.body(), kind, severity))

    @app.post("/api/reset")
    def reset():
        demo.reset()
        return {"ok": True}

    return app
