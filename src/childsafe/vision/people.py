"""Person crop transforms and landmark ownership in original-image coordinates."""
from dataclasses import dataclass
import cv2
import numpy as np
from childsafe.vision.tracking import to_original


@dataclass(frozen=True)
class PersonCrop:
    box: tuple[int,int,int,int]
    scale: float


def detect_person(detector,rgb,box,t,cfg,lone=False):
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
    return lm
