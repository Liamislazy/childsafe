"""Local web demo: compare two Module A checkpoints (e.g. standard vs A5-robust) on the same
webcam frames, each with its own alert processor, plus a photo-upload tester.

Privacy: the server listens on 127.0.0.1 only, frames are processed in memory and never
written to disk. Run with scripts/web_demo.py.
"""
from __future__ import annotations

import base64
import asyncio
import io
import math
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image,UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from childsafe.alerts.processor import LEVEL_NAMES, AlertProcessor
from childsafe.data.corruptions import KINDS, corrupt
from childsafe.vision.geometry import MOUTH, GeometryExtractor
from childsafe.vision.landmarks import LandmarkDetector
from childsafe.vision.object_classifier import ObjectClassifier, hand_crop_box
from childsafe.vision.elevated import ElevatedQuality
from childsafe.pipeline import FramePipeline
from childsafe.vision.tracking import iou
from childsafe.runtime.browser import BrowserLease, FrameRejected
from childsafe.web.labels import reason_vi
from childsafe.evaluation.sessions import file_hash,config_hash,code_hash

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
    with Image.open(io.BytesIO(body)) as image:
        if image.width*image.height > 12_000_000:
            raise ValueError('decoded image exceeds pixel limit')
        return np.asarray(image.convert("RGB"))


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


class RuntimeDemo(Demo):
    """One full multi-person runtime for live camera frames (the photo-test tab was removed)."""
    def __init__(self, cfg, models):
        if not models:
            raise ValueError('A live checkpoint is required')
        self.cfg, self.lock = cfg, threading.Lock()
        self.live_name = next(iter(models))            # only the live model is loaded
        self.paths = {self.live_name: models[self.live_name]}
        self.classifiers = {self.live_name: ObjectClassifier(self.paths[self.live_name])}
        self.pipeline = FramePipeline(cfg, None, classifier=self.classifiers[self.live_name])
        self.provenance = {'config_sha256':config_hash(cfg),'code_sha256':code_hash(),'classifier_sha256':file_hash(models[self.live_name]),
                           'detector_sha256':file_hash(cfg['detector']['weights']) if Path(cfg['detector']['weights']).is_file() else None}
        self.pipeline.provenance = self.provenance
        self.t0 = time.perf_counter()
        self.rows = []
        self.annotations, self.open_annotation = [], None
        self.recording_full = False
        self.prev_frame_t = None

    def reset(self):
        self.pipeline.reset()
        self.rows = []
        self.annotations, self.open_annotation = [], None
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

    def frame(self, body, kind, severity, mode, *, deadline=None, frame_age_s=0.,
              sequence=None,capture_t=None,clock_uncertainty_s=None):
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
            scene, _, out = self.pipeline.process(rgb, t, frame_age_s=frame_age_s,
                                                 processing_deadline=deadline)
            for row in out['feature_rows']:
                row.update(browser_sequence=sequence,browser_capture_t=capture_t,
                           clock_uncertainty_s=clock_uncertainty_s)
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
                    'reason': person['reason'], 'reasons': [reason_vi(r) for r in person['observation_reasons']],
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
                    'd': _nan_to_none(person['d']), 'd_source': features.get('distance_source', 'hand'),
                    'd_hand': _nan_to_none(features.get('d_hand', math.nan)),
                    'd_object': _nan_to_none(features.get('d_object', math.nan)),
                    'near': bool(person['near']),
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
            return {'t':t,'runtime_session_id':self.pipeline.session_id,
                    'width':rgb.shape[1], 'height':rgb.shape[0], 'people':people, 'objects':objects,
                    'level':out['level'], 'level_name':LEVEL_NAMES[out['level']],
                    'green':out['green'], 'observation':out['observation'],
                    'reason':reason_vi(out['reason']), 'limitations':limitations,
                    'notify':any(p['notify'] for p in people),
                    'detector_ok':detector.ok, 'classifier_ok':self.pipeline.classifier is not None and not self.pipeline.classifier_error,
                    'ms':round(out['inference_s']*1000,1), 'frame_gap_ms':round(frame_gap*1000,1),
                    'max_gap_ms':round(max_gap*1000,1), 'live_model':self.live_name}


def create_app(demo: Demo) -> FastAPI:
    app = FastAPI(title="childsafe demo")
    processing = asyncio.Lock()
    lease = BrowserLease(demo.cfg.get('runtime',{}).get('stale_frame_s',.25), clock=lambda:time.perf_counter())

    @app.post('/api/camera/start')
    async def camera_start(client_t: float):
        if processing.locked():
            return JSONResponse({'error':'camera busy'},status_code=503)
        try:
            result = lease.start(client_t)
            await run_in_threadpool(demo.reset)
            demo.session_start_s=time.perf_counter()-getattr(demo,'t0',time.perf_counter())
            demo.session_end_s=None
            result.update(t=demo.session_start_s,runtime_session_id=getattr(getattr(demo,'pipeline',None),'session_id',None))
            return result
        except FrameRejected as exc:
            return JSONResponse({'error':str(exc)},status_code=409)

    @app.post('/api/camera/sync')
    async def camera_sync(token: str, client_t: float):
        try:
            return lease.sync(token,client_t)
        except FrameRejected as exc:
            return JSONResponse({'error':str(exc)},status_code=409)

    @app.post('/api/camera/end')
    async def camera_end(token: str):
        try:
            lease.end(token)
            demo.session_end_s=time.perf_counter()-getattr(demo,'t0',time.perf_counter())
            return {'ok':True}
        except FrameRejected as exc:
            return JSONResponse({'error':str(exc)},status_code=409)

    @app.post('/api/camera/renew')
    async def camera_renew(token: str, client_t: float):
        if processing.locked():
            return JSONResponse({'error':'camera busy'},status_code=503)
        try:
            return lease.renew(token,client_t)
        except FrameRejected as exc:
            return JSONResponse({'error':str(exc)},status_code=409)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {"models": {n: Path(p).parent.name for n, p in demo.paths.items()}, "corruptions": KINDS,
                'runtime': isinstance(demo, RuntimeDemo), 'live_model':getattr(demo,'live_name',None),
                'stale_frame_s':demo.cfg.get('runtime',{}).get('stale_frame_s',.25),
                'provenance':getattr(demo,'provenance',{}),
                'assessment_scope':'Vật gần vùng miệng trên ảnh; chưa nhận diện nghẹn',
                "thresholds": {k: demo.cfg["temporal"][k] for k in ("d_enter", "d_exit")}}

    @app.post("/api/frame")
    async def frame(request: Request, kind: str = "", severity: int = 0, mode: str = "hand",
                    token: str = '', sequence: int = -1, capture_t: float = float('nan')):
        # Inference must not block the event loop (/info/reset can remain responsive).
        # Reject overlap rather than queuing old images behind the current frame.
        if processing.locked():
            return JSONResponse({'error': 'frame processing busy; retry with a fresh frame'},
                                status_code=503, headers={'Retry-After': '1'})
        async with processing:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 8*1024*1024:
                    return JSONResponse({'error':'frame exceeds 8 MB limit'},status_code=413)
            body = bytes(body)
            if not isinstance(demo,RuntimeDemo):
                return JSONResponse(await run_in_threadpool(demo.frame, body, kind, severity, mode))
            try:
                deadline, age = lease.frame(token,sequence,capture_t)
                result = await run_in_threadpool(demo.frame, body, kind, severity, mode,
                                                deadline=deadline,frame_age_s=age,sequence=sequence,
                                                capture_t=capture_t,clock_uncertainty_s=lease.uncertainty)
                lease.check(token)  # Stop/expiry while inference ran discards the reply.
                result['clock_uncertainty_s'] = lease.uncertainty
                result['age_scope'] = 'browser snapshot to server; not camera sensor latency'
                return JSONResponse(result)
            except FrameRejected as exc:
                return JSONResponse({'error':str(exc)},status_code=409)
            except (UnidentifiedImageError,ValueError,OSError):
                return JSONResponse({'error':'invalid image or image exceeds pixel limit'},status_code=422)

    @app.post("/api/reset")
    async def reset(token: str = ''):
        if processing.locked():
            return JSONResponse({'error':'camera busy; retry reset'},status_code=503)
        if isinstance(demo,RuntimeDemo):
            try:
                lease.check(token)
                lease.reset_floor = lease.clock()
            except FrameRejected as exc:
                return JSONResponse({'error':str(exc)},status_code=409)
        with demo.lock:
            demo.reset()
            if isinstance(demo,RuntimeDemo):
                demo.session_start_s=time.perf_counter()-demo.t0
                demo.session_end_s=None
        return {"ok": True}

    @app.get('/api/session.csv')
    def session_csv():
        if not isinstance(demo, RuntimeDemo):
            return Response(status_code=404)
        return Response(demo.session_csv(), media_type='text/csv',
                        headers={'Content-Disposition':'attachment; filename="camera_session.csv"'})

    @app.post('/api/annotation')
    async def annotation(token: str, action: str, label: str='uncertain'):
        if not isinstance(demo,RuntimeDemo):
            return Response(status_code=404)
        try:
            lease.check(token)
            if action not in ('start','end') or label not in ('unsafe_mouthing','non_event','uncertain'):
                raise FrameRejected('invalid annotation action/label')
            now=time.perf_counter()-demo.t0
            if action=='start':
                if demo.open_annotation is not None:
                    raise FrameRejected('finish the current annotation first')
                demo.open_annotation={'start':now,'label':label,'session_id':demo.pipeline.session_id}
            else:
                if demo.open_annotation is None:
                    raise FrameRejected('no open annotation')
                if len(demo.annotations)>=10000:
                    raise FrameRejected('annotation capacity reached; export and reset')
                demo.annotations.append(dict(demo.open_annotation,end=now))
                demo.open_annotation=None
            return {'t':now,'action':action,'scope':'observer click received by server; annotation delay not measured'}
        except FrameRejected as exc:
            return JSONResponse({'error':str(exc)},status_code=409)

    @app.get('/api/annotations.csv')
    async def annotations():
        if not isinstance(demo,RuntimeDemo):
            return Response(status_code=404)
        import pandas as pd
        return Response(pd.DataFrame(demo.annotations,columns=['start','end','label','session_id']).to_csv(index=False),
                        media_type='text/csv',headers={'Content-Disposition':'attachment; filename="annotations.csv"'})

    @app.get('/api/session.json')
    async def session_metadata():
        if not isinstance(demo,RuntimeDemo):
            return Response(status_code=404)
        return {'provenance':getattr(demo,'provenance',{}),'start_s':getattr(demo,'session_start_s',None),
                'end_s':getattr(demo,'session_end_s',None),'runtime_session_id':demo.pipeline.session_id,
                'unfinished_annotation':getattr(demo,'open_annotation',None) is not None,
                'annotation_clock':'observer click arrival on server; human reaction delay not measured',
                'scope':'numerical metadata only; no sensor capture timestamp or medical diagnosis'}

    return app
