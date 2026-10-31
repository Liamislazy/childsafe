"""Person crop transforms and landmark ownership in original-image coordinates."""
from dataclasses import dataclass
import cv2
import numpy as np
import time
from childsafe.vision.tracking import to_original


@dataclass(frozen=True)
class PersonCrop:
    box: tuple[int,int,int,int]
    scale: float


def detect_person(detector,rgb,box,t,cfg,lone=False,deadline=None):
    h,w=rgb.shape[:2]; bw,bh=box[2]-box[0],box[3]-box[1]
    far=cfg.get('zoom',True) and bh<cfg.get('zoom_below_frac',.6)*h
    if lone and not far:
        crop=(0,0,w,h)
    elif far:
        mx,my=int(bw*.15),int(bh*.1)
        crop=(max(0,box[0]-mx),max(0,box[1]-my),min(w,box[2]+mx),min(h,box[3]+my))
    else: crop=box
    image=rgb[crop[1]:crop[3],crop[0]:crop[2]]; scale=1.
    longest=max(image.shape[:2])
    if far and 0<longest<cfg.get('zoom_target_px',640):
        scale=min(cfg.get('zoom_target_px',640)/longest,cfg.get('zoom_max_scale',3.))
        image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
    transform=PersonCrop(crop,scale)
    lm=to_original(detector.detect(np.ascontiguousarray(image),t),transform.box,w,h,transform.scale)
    def inside(points):
        return box[0]<=points[:,0].mean()<=box[2] and box[1]<=points[:,1].mean()<=box[3]
    if lm.face is not None and not inside(lm.face): lm.face=None
    owned=[i for i,hand in enumerate(lm.hands) if inside(hand)]
    lm.hands=[lm.hands[i] for i in owned]
    if hasattr(lm,'handedness'):
        lm.handedness=[lm.handedness[i] for i in owned if i<len(lm.handedness)]
    lm.hand_roi_attempted=False; lm.hand_roi_found=0; lm.hand_roi_error=''
    if (cfg.get('hand_roi_retry',False) and not lm.hands and lm.face is not None
            and hasattr(detector,'detect_hands')
            and (deadline is None or deadline-time.perf_counter()>=cfg.get('hand_roi_budget_s',.045))):
        from childsafe.vision.geometry import face_measures
        fm=face_measures(lm.face); fw=fm['face_width']
        if fw>=cfg.get('hand_roi_min_face_px',40):
            side=min(max(fw*cfg.get('hand_roi_scale',3.),cfg.get('hand_roi_min_px',128)),min(w,h))
            if side*side<=cfg.get('hand_roi_max_frame_frac',.35)*w*h:
                center=fm['mouth']+np.array([0.,fw*.35])
                rx=int(min(max(center[0]-side/2,0),w-side))
                ry=int(min(max(center[1]-side/2,0),h-side))
                roi=(rx,ry,int(rx+side),int(ry+side))
                lm.hand_roi_attempted=True; lm.hand_roi_box=roi
                try:
                    retry=to_original(detector.detect_hands(np.ascontiguousarray(rgb[roi[1]:roi[3],roi[0]:roi[2]]),t),roi,w,h)
                    owned=[i for i,hand in enumerate(retry.hands) if inside(hand)]
                    lm.hands=[retry.hands[i] for i in owned]
                    lm.handedness=[retry.handedness[i] for i in owned if i<len(retry.handedness)]
                    lm.hand_roi_found=len(lm.hands)
                except Exception as exc:
                    lm.hand_roi_error=type(exc).__name__  # keep the primary face result
    return lm
