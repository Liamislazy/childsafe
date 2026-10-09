"""Conservative image-plane proximity and uncalibrated elevated-view quality gates."""
from __future__ import annotations

import math
import numpy as np

MOUTH_POLYGON = [78, 13, 308, 14]


def mask_region_distance(mask, polygon):
    """Minimum distance from visible foreground pixel centers to a convex mouth region.

    Tests/use rely only on visible masks, never a box center or an inferred object tip.
    Polygon winding may be either direction. Distances are pixels, not centimetres.
    """
    polygon = np.asarray(polygon, float)
    if mask is None or len(polygon) < 3 or not np.isfinite(polygon).all():
        return math.nan
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return math.nan
    best = math.inf
    a, b = polygon, np.roll(polygon, -1, axis=0)
    segments = b-a
    for start in range(0,len(xs),8192):
        points = np.column_stack((xs[start:start+8192],ys[start:start+8192]))
        delta = points[:,None,:]-a[None,:,:]
        cross = segments[None,:,0]*delta[:,:,1]-segments[None,:,1]*delta[:,:,0]
        inside = np.all(cross >= -1e-8,axis=1) | np.all(cross <= 1e-8,axis=1)
        if inside.any():
            return 0.
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
        gray = rgb.astype(np.float32).mean(2)
        brightness = float(gray.mean()) if gray.size else 0.
        lap = (4*gray[1:-1,1:-1]-gray[:-2,1:-1]-gray[2:,1:-1]-
               gray[1:-1,:-2]-gray[1:-1,2:])
        sharpness = float(lap.var()) if lap.size else 0.
        reasons = []
        if not qc.get('min_brightness',40) <= brightness <= qc.get('max_brightness',220):
            reasons.append('poor exposure')
        if sharpness < qc.get('min_sharpness',20):
            reasons.append('blurred image')
        visible = lm is not None and lm.face is not None
        yaw = pitch = math.nan
        if visible:
            if features.get('face_width',0) < qc.get('min_face_px',40):
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
            if np.ptp(mouth[:,0]) < qc.get('min_mouth_px',4):
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
                if hand_extent < qc.get('min_hand_px',10):
                    reasons.append('hand too small')
            else:
                reasons.append('no observable hand')
        if 'poor exposure' in reasons or 'blurred image' in reasons:
            visible = False
        return dict(mouth_visible=visible, head_yaw=yaw, head_pitch=pitch,
                    brightness=brightness, sharpness=sharpness, quality_reasons=reasons)


def object_proximity(detection, lm, f, cfg):
    mask_present = (detection is not None and detection.mask is not None and
               detection.mask.shape == (lm.height,lm.width) and
               np.count_nonzero(detection.mask) >= cfg.get('min_mask_px',16)) if lm is not None else False
    mask_ok = mask_present and detection.confidence >= cfg.get('min_mask_conf',.6)
    observable = (f.get('mouth_visible',False) and f.get('hand_ok',False) and
                  not f.get('association_ambiguous',False) and
                  'hand too small' not in f.get('quality_reasons',[]))
    trusted = (mask_ok and f.get('mouth_visible',False) and f.get('hand_ok',False) and
               not f.get('association_ambiguous',False) and 'hand too small' not in f.get('quality_reasons',[]))
    d_px = mask_region_distance(detection.mask,lm.face[MOUTH_POLYGON,:2]) if trusted else math.nan
    # Low-confidence localization can request attention, but never supplies
    # confirmed evidence or dwell toward a danger alert.
    candidate = (mask_present and observable and detection.confidence >= cfg.get('min_candidate_mask_conf',.5))
    candidate_px = mask_region_distance(detection.mask,lm.face[MOUTH_POLYGON,:2]) if candidate else math.nan
    return dict(mask_reliable=mask_ok, proximity_evidence=bool(trusted), object_d_px=d_px,
                object_d_relative=d_px/max(f.get('face_width',1),1) if trusted else math.nan,
                candidate_object_d_relative=candidate_px/max(f.get('face_width',1),1) if candidate else math.nan,
                object_label=detection.label if detection is not None else '',
                object_conf=detection.confidence if detection is not None else math.nan)
