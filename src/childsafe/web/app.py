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
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from childsafe.alerts.processor import LEVEL_NAMES, AlertProcessor
from childsafe.data.corruptions import KINDS, corrupt
from childsafe.vision.geometry import MOUTH, GeometryExtractor
from childsafe.vision.landmarks import LandmarkDetector
from childsafe.vision.object_classifier import ObjectClassifier, hand_crop_box
from childsafe.vision.elevated import ElevatedQuality
from childsafe.pipeline import FramePipeline
from childsafe.vision.tracking import iou

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


def _displayed_people(persons):
    """People detected in this frame, plus held IDs (briefly missed) only when no detected person
    is at the same place. Otherwise a person who got a new ID was shown twice ("ghost" cards)."""
    seen = [p for p in persons if p.get('visible', True)]
    def near(held, other):
        hb, ob = held.get('box'), other.get('box')
        if hb is None or ob is None:
            return True
        cx, cy = (ob[0]+ob[2])/2, (ob[1]+ob[3])/2
        return iou(hb, ob) > 0 or (hb[0] <= cx <= hb[2] and hb[1] <= cy <= hb[3])
    held = [p for p in persons if not p.get('visible', True) and not any(near(p, o) for o in seen)]
    return seen + held


def _prediction_status(obj, threshold):
    if obj is None:
        return 'none'
    p = obj['p_unsafe_cal']
    if (not math.isfinite(p) or not math.isfinite(obj.get('obj_conf',math.nan)) or
            max(p,1-p) < threshold or obj['obj_conf'] < threshold or
            obj.get('obj_class') in ('other','unknown','empty_hand') or obj.get('p_unknown_cal',0) >= .5):
        return 'uncertain'
    return 'unsafe' if p >= .5 else 'safe'


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
        self.quality = ElevatedQuality(self.cfg.get('quality',{}))
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
            f.update(self.quality.measure(rgb,lm,f))
            # This comparator has landmarks/classifiers but no instance detector or
            # per-person associations. Never treat its hand crops as contact evidence.
            f.update(schema_version=2,detector_ok=False,classifier_ok=True,calibrated=False,
                     object_coverage=False,proximity_evidence=False)
            f['quality_reasons'].append('classifier comparison only; no object-mask runtime')
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
                    'level':res['level'],'level_name':LEVEL_NAMES[res['level']],
                    'reason':'; '.join(x for x in [res['reason'],*res['observation_reasons']] if x),
                    "d": _nan_to_none(res["d"]), "near": bool(res["near"]), "dwell": res["dwell"],
                    'lost_near':bool(res['lost_near']),
                    'obj_status':_prediction_status(obj,self.cfg['calibration']['reject_threshold']),
                    'observation':res['observation'],'green':res['green'],
                    'observation_reasons':res['observation_reasons'],
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
                               'status':_prediction_status(r,self.cfg['calibration']['reject_threshold'])}
                           for n, r in res.items()}}


class RuntimeDemo(Demo):
    """One full multi-person runtime for live frames; photo comparison stays separate."""
    def __init__(self, cfg, models):
        if not models:
            raise ValueError('A live checkpoint is required')
        self.cfg, self.lock = cfg, threading.Lock()
        self.paths = models
        self.classifiers = {name: ObjectClassifier(path) for name, path in models.items()}
        self.live_name = next(iter(models))
        self.pipeline = FramePipeline(cfg, None, classifier=self.classifiers[self.live_name])
        self.t0 = time.perf_counter()
        self.rows = []
        self.recording_full = False
        self.prev_frame_t = None

    def reset(self):
        self.pipeline.reset()
        self.rows = []
        self.recording_full = False
        self.prev_frame_t = None
        self.slots = {}

    def _slot(self, shown_ids):
        """Short display number per person (1, 2, ...). Internal track IDs never repeat (alert
        history must not pass between people), so they keep climbing; the page shows a slot that is
        freed when the person is no longer shown and reused for the next one."""
        slots = getattr(self, 'slots', None)
        if slots is None:
            slots = self.slots = {}
        for pid in [k for k in slots if k not in shown_ids]:
            del slots[pid]
        for pid in shown_ids:
            if pid not in slots:
                used = set(slots.values())
                slots[pid] = next(n for n in range(1, len(used) + 2) if n not in used)
        return slots

    def session_csv(self):
        import pandas as pd
        with self.lock:
            return pd.DataFrame(self.rows).to_csv(index=False)

    def close(self):
        self.pipeline.close()

    def frame(self, body, kind, severity, mode):
        rgb = _decode(body)
        if kind in KINDS and severity > 0:
            rgb = corrupt(rgb, kind, severity, seed=int(time.time()*1000) % (1 << 30))
        with self.lock:
            t = time.perf_counter()-self.t0
            # Real spacing between frames (server work + browser round trip). This, not server time
            # alone, is what the alert processor compares with temporal.max_gap_s.
            prev = getattr(self, 'prev_frame_t', None)
            frame_gap = t-prev if prev is not None else 0.
            self.prev_frame_t = t
            scene, _, out = self.pipeline.process(rgb, t)
            if not self.recording_full:
                if len(self.rows)+len(out['feature_rows']) <= 100000:
                    self.rows.extend(dict(row) for row in out['feature_rows'])
                else:
                    self.recording_full = True
            people = []
            shown = _displayed_people(out['persons'])
            slots = self._slot([p['person_id'] for p in shown])
            for person in shown:
                pid = person['person_id']
                lm = scene.people.get(pid)
                features = next((f for f in out['feature_rows'] if f['person_id'] == pid), {})
                people.append({
                    'id': slots[pid], 'track_id': pid, 'box': scene.boxes.get(pid),
                    'level': person['level'], 'level_name': LEVEL_NAMES[person['level']],
                    'observation': person['observation'], 'green': person['green'],
                    'reason': person['reason'], 'reasons': person['observation_reasons'],
                    'face': bool(lm is not None and lm.face is not None),
                    'mouth_visible': bool(features.get('mouth_visible')),
                    'hands': [h[:,:2].round(1).tolist() for h in lm.hands] if lm else [],
                    'mouth': lm.face[MOUTH,:2].mean(0).round(1).tolist() if lm is not None and lm.face is not None else None,
                    # Our classifier names the object; YOLO's COCO label is only a hint (COCO has no
                    # pen, so a pen comes back as 'toothbrush').
                    'object': person['obj_class'] or ('vật trong tay' if features.get('object_present') else ''),
                    'yolo_label': features.get('object_label', ''),
                    'mask_reliable': bool(features.get('mask_reliable')),
                    'class': person['obj_class'], 'status': person['obj_status'],
                    'p_unsafe': _nan_to_none(person['p_unsafe']),
                    'd': _nan_to_none(person['d']), 'near': bool(person['near']),
                    'dwell': person['dwell'], 'notify': bool(person['notify'])})
            objects = []
            for obj in scene.objects:
                polygon = []
                if obj.mask is not None:
                    import cv2
                    contours, _ = cv2.findContours(obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        polygon = cv2.approxPolyDP(contour, .003*cv2.arcLength(contour, True), True)[:,0,:].tolist()
                objects.append({'label':obj.label, 'confidence':obj.confidence, 'box':list(obj.box), 'mask':polygon})
            detector = self.pipeline.detector
            limitations = ['Chưa hiệu chuẩn camera; đũa, bút và đồng xu chưa có detector riêng.']
            if self.recording_full:
                limitations.append('Bộ nhớ số liệu đã đầy: tải phiên hiện tại rồi Reset alerts để bắt đầu phiên mới.')
            if not detector.ok:
                limitations.append('Detector không sẵn sàng: '+getattr(detector,'error',''))
            if not people:
                limitations.append('Chưa phát hiện người: đưa người và tay vào khung hình, tăng ánh sáng.')
            max_gap = self.cfg['temporal'].get('max_gap_s',.25)
            if frame_gap > max_gap or out['inference_s'] > max_gap:
                limitations.append(f'Khoảng giữa hai frame {frame_gap*1000:.0f} ms (xử lý {out["inference_s"]*1000:.0f} ms) '
                                   f'vượt {max_gap*1000:.0f} ms: lịch sử cảnh báo bị đặt lại nên DANGER không tích lũy được. '
                                   'Giảm số người trong khung, đóng ứng dụng nặng, hoặc dùng GPU.')
            return {'width':rgb.shape[1], 'height':rgb.shape[0], 'people':people, 'objects':objects,
                    'level':out['level'], 'level_name':LEVEL_NAMES[out['level']],
                    'green':out['green'], 'observation':out['observation'],
                    'reason':out['reason'], 'limitations':limitations,
                    'notify':any(p['notify'] for p in people),
                    'detector_ok':detector.ok, 'classifier_ok':self.pipeline.classifier is not None and not self.pipeline.classifier_error,
                    'ms':round(out['inference_s']*1000,1), 'frame_gap_ms':round(frame_gap*1000,1),
                    'max_gap_ms':round(max_gap*1000,1), 'live_model':self.live_name}


def create_app(demo: Demo) -> FastAPI:
    app = FastAPI(title="childsafe demo")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {"models": {n: Path(p).parent.name for n, p in demo.paths.items()}, "corruptions": KINDS,
                'runtime': isinstance(demo, RuntimeDemo), 'live_model':getattr(demo,'live_name',None),
                "thresholds": {k: demo.cfg["temporal"][k] for k in ("d_enter", "d_exit")}}

    @app.post("/api/frame")
    async def frame(request: Request, kind: str = "", severity: int = 0, mode: str = "hand"):
        return JSONResponse(demo.frame(await request.body(), kind, severity, mode))

    @app.post("/api/image")
    async def image(request: Request, kind: str = "", severity: int = 0):
        return JSONResponse(demo.image(await request.body(), kind, severity))

    @app.post("/api/reset")
    def reset():
        with demo.lock:
            demo.reset()
        return {"ok": True}

    @app.get('/api/session.csv')
    def session_csv():
        if not isinstance(demo, RuntimeDemo):
            return Response(status_code=404)
        return Response(demo.session_csv(), media_type='text/csv',
                        headers={'Content-Disposition':'attachment; filename="camera_session.csv"'})

    return app
