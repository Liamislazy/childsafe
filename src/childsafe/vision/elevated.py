"""Conservative image-plane proximity and uncalibrated elevated-view quality gates."""
from __future__ import annotations

import math

import cv2
import numpy as np

MOUTH_POLYGON = [78, 13, 308, 14]
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
LAPLACE = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)


def mask_region_distance(mask, polygon):
    """Minimum distance from visible foreground pixel centers to a convex mouth region.

    Tests/use rely only on visible masks, never a box center or an inferred object tip.
    Polygon winding may be either direction. Distances are pixels, not centimetres.
    """
    polygon = np.asarray(polygon, float)
    if mask is None or len(polygon) < 3 or not np.isfinite(polygon).all():
        return math.nan
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return math.nan
    a, b = polygon, np.roll(polygon, -1, axis=0)
    segments = b-a
    # 1) Only pixels inside the polygon's bounding box can lie inside the (convex) mouth region.
    px0, py0 = np.floor(polygon.min(0)).astype(int)
    px1, py1 = np.ceil(polygon.max(0)).astype(int)
    px0, py0 = max(px0, 0), max(py0, 0)
    ys, xs = np.nonzero(mask[py0:py1+1, px0:px1+1]) if px1 >= 0 and py1 >= 0 else ((), ())
    if len(xs):
        delta = np.column_stack((xs+px0, ys+py0))[:,None,:]-a[None,:,:]
        cross = segments[None,:,0]*delta[:,:,1]-segments[None,:,1]*delta[:,:,0]
        if (np.all(cross >= -1e-8,axis=1) | np.all(cross <= 1e-8,axis=1)).any():
            return 0.
    # 2) Otherwise the nearest mask pixel is on the mask boundary (an interior pixel always has a
    # neighbour closer to an outside convex region), so only boundary pixels are measured. Same
    # distances as measuring every pixel, within <1 px, at a fraction of the cost for large masks.
    y0, y1, x0, x1 = rows[0], rows[-1]+1, cols[0], cols[-1]+1
    roi = np.pad(mask[y0:y1, x0:x1], 1).astype(np.uint8)
    boundary = roi.astype(bool) & ~cv2.erode(roi, CROSS).astype(bool)
    ys, xs = np.nonzero(boundary[1:-1, 1:-1])
    best = math.inf
    for start in range(0,len(xs),8192):
        points = np.column_stack((xs[start:start+8192]+x0,ys[start:start+8192]+y0))
        delta = points[:,None,:]-a[None,:,:]
        projection = np.clip((delta*segments[None,:,:]).sum(2) /
                             np.maximum((segments*segments).sum(1),1e-9),0,1)
        distances = np.linalg.norm(delta-projection[:,:,None]*segments[None,:,:],axis=2)
        best = min(best,float(distances.min()))
    return best


class ElevatedQuality:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_mouth = self.prev_t = None

    def measure(self, rgb, lm, features):
        qc = self.cfg
        h, w = rgb.shape[:2]
        bounds = (0, 0, w, h)
        if lm is not None and lm.face is not None:
            points = lm.face[:, :2]
            if np.isfinite(points).all():
                lo, hi = np.floor(points.min(0)).astype(int), np.ceil(points.max(0)).astype(int) + 1
                bounds = (max(0, int(lo[0])), max(0, int(lo[1])),
                          min(w, int(hi[0])), min(h, int(hi[1])))
        x0, y0, x1, y1 = bounds
        roi = rgb[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else rgb[:0, :0]
        # Same channel-mean gray and 4-neighbour Laplacian variance as before, computed with
        # OpenCV (about 2x faster on a 1280x720 frame).
        gray = np.add.reduce(roi, axis=2, dtype=np.uint16).astype(np.float32) * np.float32(1/3)
        brightness = float(cv2.mean(gray)[0]) if gray.size else 0.
        if min(gray.shape[:2]) > 2:
            lap = cv2.filter2D(gray, cv2.CV_32F, LAPLACE, borderType=cv2.BORDER_REPLICATE)[1:-1,1:-1]
            sharpness = float(cv2.meanStdDev(lap)[1][0, 0] ** 2)
        else:
            sharpness = 0.
        reasons = []
        if not qc.get('min_brightness',40) <= brightness <= qc.get('max_brightness',220):
            reasons.append('poor exposure')
        if sharpness < qc.get('min_sharpness',20):
            reasons.append('blurred image')
        visible = lm is not None and lm.face is not None
        yaw = pitch = math.nan
        face_px = float(features.get('face_width', 0))
        mouth_px = hand_px = 0.
        zoom = float(getattr(lm, 'scale', 1.) or 1.) if lm is not None else 1.
        if visible:
            # Coordinates have already been mapped to original pixels. Zoom adds no detail.
            if face_px < qc.get('min_face_px',40):
                reasons.append('face too small')
                visible = False
            matrix = getattr(lm,'face_matrix',None)
            if matrix is None or np.shape(matrix) != (4,4) or not np.isfinite(matrix).all():
                reasons.append('head orientation unavailable')
                visible = False
            else:
                rotation = np.asarray(matrix)[:3,:3]
                # Remove any uniform scale; these are camera-image orientation proxies.
                rotation = rotation / np.maximum(np.linalg.norm(rotation,axis=0),1e-9)
                yaw = math.degrees(math.atan2(-rotation[2,0], math.hypot(rotation[0,0],rotation[1,0])))
                pitch = math.degrees(math.atan2(rotation[2,1],rotation[2,2]))
                if abs(yaw) > qc.get('max_yaw_deg',60) or abs(pitch) > qc.get('max_pitch_deg',45):
                    reasons.append('mouth hidden by head orientation')
                    visible = False
            mouth = lm.face[MOUTH_POLYGON,:2]
            mouth_px = float(np.ptp(mouth[:,0]))
            if mouth_px < qc.get('min_mouth_px',4):
                reasons.append('mouth too small')
                visible = False
            current = mouth.mean(0)
            if self.prev_mouth is not None and 0 < lm.t-self.prev_t <= .25:
                jitter = float(np.linalg.norm(current-self.prev_mouth)/max(features.get('face_width',1),1))
                if jitter > qc.get('max_landmark_jump',.35):
                    reasons.append('unstable mouth landmarks')
                    visible = False
            self.prev_mouth,self.prev_t = current,lm.t
            if lm.hands:
                hand_extent = max(float(np.ptp(hand[:,:2],axis=0).max()) for hand in lm.hands)
                hand_px = hand_extent
                if hand_extent < qc.get('min_hand_px',10):
                    reasons.append('hand too small')
            else:
                reasons.append('no observable hand')
        if 'poor exposure' in reasons or 'blurred image' in reasons:
            visible = False
        return dict(mouth_visible=visible, head_yaw=yaw, head_pitch=pitch,
                    brightness=brightness, sharpness=sharpness, quality_reasons=reasons,
                    quality_schema_version=3, quality_roi_x0=x0, quality_roi_y0=y0,
                    quality_roi_x1=x1, quality_roi_y1=y1, landmark_scale=zoom,
                    original_face_px=face_px, original_mouth_px=mouth_px, original_hand_px=hand_px)


def object_proximity(detection, lm, f, cfg):
    mask = detection.mask if detection is not None else None
    from_box = False
    if detection is not None and mask is None and lm is not None:
        # No segmentation mask: use the object's box, so the distance is still object -> mouth.
        # A box is never trusted as confirmed (DANGER) evidence, only as a candidate.
        mask = np.zeros((lm.height, lm.width), bool)
        x0, y0, x1, y1 = (int(round(v)) for v in detection.box)
        mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        from_box = True
    mask_present = (mask is not None and mask.shape == (lm.height,lm.width) and
                    np.count_nonzero(mask) >= cfg.get('min_mask_px',16)) if lm is not None else False
    mask_ok = mask_present and not from_box and detection.confidence >= cfg.get('min_mask_conf',.6)
    observable = (f.get('mouth_visible',False) and f.get('hand_ok',False) and
                  not f.get('association_ambiguous',False) and
                  'hand too small' not in f.get('quality_reasons',[]))
    trusted = (mask_ok and f.get('mouth_visible',False) and f.get('hand_ok',False) and
               not f.get('association_ambiguous',False) and 'hand too small' not in f.get('quality_reasons',[]))
    # Low-confidence localization can request attention, but never supplies
    # confirmed evidence or dwell toward a danger alert.
    candidate = (mask_present and observable and detection.confidence >= cfg.get('min_candidate_mask_conf',.5))
    # trusted implies candidate whenever min_candidate_mask_conf <= min_mask_conf: measure once.
    measured = (mask_region_distance(mask,lm.face[MOUTH_POLYGON,:2])
                if trusted or candidate else math.nan)
    d_px = measured if trusted else math.nan
    candidate_px = measured if candidate else math.nan
    return dict(mask_reliable=mask_ok, proximity_evidence=bool(trusted), object_d_px=d_px,
                object_d_relative=d_px/max(f.get('face_width',1),1) if trusted else math.nan,
                candidate_object_d_relative=candidate_px/max(f.get('face_width',1),1) if candidate else math.nan,
                object_label=detection.label if detection is not None else '',
                object_conf=detection.confidence if detection is not None else math.nan)
