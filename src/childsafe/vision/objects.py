"""Per-person object assignment and temporal mask evidence, without physical-distance claims."""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from childsafe.vision.tracking import iou,object_candidates


def same_object(prev_box,prev_label,obj):
    if prev_label!=obj.label: return False
    if iou(prev_box,obj.box)>=.2: return True
    a=np.asarray(prev_box); b=np.asarray(obj.box)
    size=max(a[2]-a[0],a[3]-a[1])
    return np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)<=size


@dataclass
class ObjectAssignments:
    owned: dict
    ambiguous: set


def assign_objects(objects,people,landmarks,max_distance_ratio=1.5):
    result=ObjectAssignments({p.person_id:[] for p in people},set())
    for obj in objects:
        ids=object_candidates(obj,people,landmarks,max_distance_ratio=max_distance_ratio)
        if len(ids)==1: result.owned[ids[0]].append(obj)
        elif ids: result.ambiguous.update(ids)
    return result


class MaskEvidence:
    def __init__(self,cfg,max_gap_s=.25):
        self.cfg,self.max_gap_s=cfg,max_gap_s
        self.reset()

    def reset(self):
        self.previous=None; self.observed_s=0.; self.missing=False

    def apply(self,obj,t,proximity):
        result=dict(proximity); stable=False
        if obj is None and self.previous is not None and 0<t-self.previous[0]<=self.max_gap_s:
            self.missing=True  # pause: never add an unobserved interval to stability
        elif obj is None or not result.get('mask_reliable'):
            self.reset()
        else:
            ys,xs=np.nonzero(obj.mask)
            crop=obj.mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
            signature=cv2.resize(crop.astype(np.uint8),(32,32),interpolation=cv2.INTER_NEAREST).astype(bool)
            area=int(obj.mask.sum())
            if self.previous is not None:
                pt,box,label,old_signature,old_area=self.previous
                overlap=np.count_nonzero(signature&old_signature)/max(np.count_nonzero(signature|old_signature),1)
                continuous=(0<t-pt<=self.max_gap_s and same_object(box,label,obj) and
                            max(area,old_area)/max(min(area,old_area),1)<=self.cfg.get('max_mask_area_ratio',2.) and
                            overlap>=self.cfg.get('min_mask_shape_iou',.5))
                if not continuous:
                    self.observed_s=0.
                elif not self.missing:
                    self.observed_s+=t-pt
            else: self.observed_s=0.
            self.previous=(t,obj.box,obj.label,signature,area)
            self.missing=False
            stable=self.observed_s+1e-9>=self.cfg.get('mask_stability_s',.1)
        result['mask_temporally_stable']=stable
        if not stable:
            result.update(proximity_evidence=False,object_d_px=math.nan,object_d_relative=math.nan)
        return result
