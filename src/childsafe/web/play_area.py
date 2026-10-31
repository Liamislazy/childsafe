"""Fixed-view preparation check: visible candidates only, never a safety certificate.

No face/hand analysis, classifier, implicit downloads, image persistence or medical
inference. The detector is injectable so API/quality behavior can be tested without
pretending synthetic fixtures demonstrate camera accuracy.
"""
from __future__ import annotations

import asyncio
import io
import math
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from childsafe.vision.detection import SegmentationDetector, select_weights

STATIC = Path(__file__).parent / 'static'
NO_CACHE = {'Cache-Control': 'no-store'}
LABELS = {'knife': 'Dao', 'scissors': 'Kéo', 'fork': 'Nĩa'}
LIMIT = 'Chỉ gợi ý dao, kéo, nĩa nhìn thấy trong vùng rà soát; có thể nhận nhầm hoặc bỏ sót.'
OUTSIDE = 'Bút, đồng xu, pin, ổ điện, máy sưởi và kích thước/tầm với của trẻ chưa được đánh giá.'
MAX_BODY = 8 * 1024 * 1024
MAX_PIXELS = 6_000_000


def read_rgb(body: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(body)) as image:
        if image.format not in {'JPEG', 'PNG'} or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Chỉ dùng ảnh JPEG hoặc PNG tĩnh.')
        w, h = image.size
        if w*h > MAX_PIXELS or min(w, h) < 64 or max(w, h) > 4096:
            raise ValueError('Ảnh cần từ 64 px mỗi cạnh, tối đa 6 triệu pixel và 4096 px mỗi cạnh.')
        # Respect camera/file orientation before defining the scan rectangle.
        from PIL import ImageOps
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()


def image_quality(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = min(1., 640/max(w, h))
    sample = cv2.resize(gray, (max(1, round(w*scale)), max(1, round(h*scale))))
    sharpness = float(cv2.Laplacian(sample, cv2.CV_64F).var())
    mean = float(gray.mean())
    dark = float((gray < 20).mean())
    bright = float((gray > 245).mean())
    reasons = []
    if min(w, h) < 240:
        reasons.append('Vùng rà soát quá nhỏ; đưa camera gần hoặc chọn vùng lớn hơn.')
    if mean < 35 or dark > .7:
        reasons.append('Ảnh quá tối; tăng ánh sáng và chụp lại.')
    if mean > 235 or bright > .7:
        reasons.append('Ảnh bị sáng lóa; đổi góc hoặc ánh sáng và chụp lại.')
    if sharpness < 35:
        reasons.append('Ảnh thiếu chi tiết hoặc nhòe; giữ camera cố định và chụp lại.')
    return dict(usable=not reasons, reasons=reasons, sharpness=round(sharpness, 2),
                mean=round(mean, 2), thresholds_validated=False)


class PlayAreaScanner:
    def __init__(self, cfg: dict, detector=None):
        self.cfg, self.detector = cfg, detector
        self.lock = threading.Lock()

    def _load(self):
        if self.detector is None:
            dc = self.cfg.get('detector', {})
            self.detector = SegmentationDetector(select_weights(dc), confidence=.35,
                object_confidence=.35, device=dc.get('device', 'auto'), imgsz=640,
                cpu_threads=self.cfg.get('runtime', {}).get('cpu_threads', 4))
        return self.detector

    def info(self):
        with self.lock:
            detector = self._load()
            return dict(detector_ok=detector.ok,
                model=getattr(detector, 'weights_file', 'test-injected'),
                model_sha256=getattr(detector, 'checksum', ''),
                device=str(getattr(detector, 'device', 'unknown')),
                supported=[dict(label=k, name=v) for k, v in LABELS.items() if k in detector.names],
                scope=LIMIT, outside_scope=OUTSIDE, calibrated=False,
                images_saved=False, object_confidence_threshold=.35,
                thresholds=dict(min_crop_px=240, min_sharpness=35, min_mean=35, max_mean=235,
                                max_dark_fraction=.7, max_bright_fraction=.7))

    def scan(self, body: bytes, inset: float):
        started = time.perf_counter()
        rgb = read_rgb(body)
        height, width = rgb.shape[:2]
        x, y = round(width*inset), round(height*inset)
        roi = [x, y, width-x, height-y]
        crop = rgb[y:height-y, x:width-x].copy()
        quality = image_quality(crop)
        candidates = []
        with self.lock:
            detector = self._load()
            prediction_failed = False
            if detector.ok or getattr(detector, 'model', None) is not None:
                # Detect even on poor images, but never use them as evidence of absence.
                try:
                    detections = detector.detect(crop)
                except Exception:
                    detections, prediction_failed = [], True
            else:
                detections = []
            ok = bool(detector.ok) and not prediction_failed
            names = set(detector.names)
            for d in detections:
                if d.label not in LABELS or d.label not in names:
                    continue
                b = np.asarray(d.box, dtype=float)
                if b.shape != (4,) or not np.isfinite(b).all() or not math.isfinite(d.confidence) or not .35 <= d.confidence <= 1:
                    continue
                a, c = np.clip(b[[0, 2]], 0, crop.shape[1])
                t, bottom = np.clip(b[[1, 3]], 0, crop.shape[0])
                if c <= a or bottom <= t:
                    continue
                box = [float(a+x), float(t+y), float(c+x), float(bottom+y)]
                # Defensive duplicate suppression; no merging two adjacent same-class objects.
                from childsafe.vision.tracking import iou
                if any(old['label'] == d.label and iou(box, old['box']) > .8 for old in candidates):
                    continue
                candidates.append(dict(label=d.label, name=LABELS[d.label], box=box,
                    confidence=round(float(d.confidence), 4),
                    at_edge=bool(a < 3 or t < 3 or c > crop.shape[1]-3 or bottom > crop.shape[0]-3),
                    action='Kiểm tra vật và cất hoặc thay bằng đồ phù hợp trước khi trẻ tiếp cận.'))
            model = getattr(detector, 'weights_file', '')
            checksum = getattr(detector, 'checksum', '')
        if not set(LABELS).issubset(names):
            quality['reasons'].append('Detector không có đủ ba lớp dao, kéo, nĩa của bản thử.')
            quality['usable'] = False
        usable = quality['usable'] and ok
        return dict(width=width, height=height, roi=roi, candidates=candidates[:40],
            quality=quality, detector_ok=ok, usable_for_comparison=usable,
            status='needs_review' if candidates else 'no_candidates' if usable else 'unavailable',
            scope=LIMIT, outside_scope=OUTSIDE, calibrated=False, model=model, model_sha256=checksum,
            ms=round((time.perf_counter()-started)*1000), images_saved=False,
            message=('Cần người lớn kiểm tra các vật được đánh dấu.' if candidates else
                     'Chưa ghi nhận dao/kéo/nĩa trong ảnh; vẫn cần kiểm tra thủ công.' if usable else
                     'Chưa đủ thông tin để so sánh; giữ nguyên các mục cần kiểm tra trước đó.'))


def play_area_router(scanner: PlayAreaScanner):
    router = APIRouter()
    processing = asyncio.Lock()

    @router.get('/play-area')
    def page():
        return FileResponse(STATIC / 'play_area.html', headers=NO_CACHE)

    @router.get('/play-area.js')
    def javascript():
        return FileResponse(STATIC / 'play_area.js', media_type='text/javascript', headers=NO_CACHE)

    @router.get('/api/play-area/health')
    def health():
        # Startup/liveness probe must not wait for model loading or inference.
        return JSONResponse({'service': 'childsafe-play-area', 'status': 'running'}, headers=NO_CACHE)

    @router.get('/api/play-area/info')
    async def info():
        return await run_in_threadpool(scanner.info)

    @router.post('/api/play-area/scan')
    async def scan(request: Request, inset: float = Query(.08, ge=0, le=.3)):
        # Require a non-simple browser request and reject cross-site origins.
        origin = request.headers.get('origin')
        if (request.headers.get('x-childsafe-scan') != '1' or
            request.headers.get('sec-fetch-site') == 'cross-site' or
            (origin and urlsplit(origin).netloc != request.url.netloc)):
            return JSONResponse({'error': 'Yêu cầu phải đến từ trang kiểm tra cục bộ.'}, status_code=403)
        if processing.locked():
            return JSONResponse({'error': 'Đang phân tích ảnh trước; hãy chụp ảnh mới sau khi xong.'}, status_code=503)
        async with processing:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY:
                    return JSONResponse({'error': 'Ảnh vượt giới hạn 8 MB.'}, status_code=413)
            try:
                result = await run_in_threadpool(scanner.scan, bytes(body), inset)
                return JSONResponse(result)
            except (UnidentifiedImageError, Image.DecompressionBombError, ValueError, OSError):
                return JSONResponse({'error': 'Ảnh không hợp lệ; dùng JPEG/PNG tối đa 6 triệu pixel.'}, status_code=422)
    return router


def create_play_area_app(cfg, detector=None):
    app = FastAPI(title='ChildSafe — chuẩn bị góc chơi')
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])
    app.include_router(play_area_router(PlayAreaScanner(cfg, detector)))

    @app.get('/')
    def home():
        return FileResponse(STATIC / 'play_area.html', headers=NO_CACHE)
    return app
