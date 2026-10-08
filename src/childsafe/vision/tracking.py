"""Conservative temporary IDs; ambiguity always discards identity history."""
from dataclasses import dataclass

import numpy as np


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    area = lambda x: max(0, x[2]-x[0]) * max(0, x[3]-x[1])
    return intersection / max(area(a)+area(b)-intersection, 1e-9)


@dataclass
class Track:
    person_id: int
    box: tuple
    last_t: float
    ambiguous: bool = False


class PersonTracker:
    def __init__(self, max_people=4, hold_s=.8, min_iou=.25, margin=.15):
        self.max_people, self.hold_s = max_people, hold_s
        self.min_iou, self.margin = min_iou, margin
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()  # monotonically increasing IDs avoid reuse across source sessions

    def update(self, detections, t):
        people = [d for d in detections if d.label == 'person']
        overflow = len(people) > self.max_people
        people = sorted(people, key=lambda d: d.confidence, reverse=True)[:self.max_people]
        old = {k: v for k, v in self.tracks.items() if t-v.last_t <= self.hold_s}
        scores = {i: sorted([(iou(d.box, v.box), k) for k, v in old.items()], reverse=True)
                  for i, d in enumerate(people)}
        choices = {i: s[0][1] for i, s in scores.items() if s and s[0][0] >= self.min_iou}
        active = []
        retired = set(self.tracks)-set(old)
        for i, d in enumerate(people):
            s = scores[i]
            overlap = any(i != j and iou(d.box, other.box) > .1 for j, other in enumerate(people))
            ambiguous = (overflow or overlap or
                         (len(s) > 1 and s[0][0] >= self.min_iou and s[0][0]-s[1][0] < self.margin) or
                         (i in choices and list(choices.values()).count(choices[i]) > 1))
            if ambiguous:
                retired.update(k for score, k in s if score >= self.min_iou)
            pid = choices.get(i) if not ambiguous else None
            if pid is None:
                pid = self.next_id
                self.next_id += 1
            track = Track(pid, d.box, t, ambiguous)
            active.append(track)
        current_ids = {v.person_id for v in active}
        retired -= current_ids
        self.tracks = {k: v for k, v in old.items() if k not in retired and k not in current_ids}
        self.tracks.update({v.person_id: v for v in active})
        return active, retired


def to_original(lm, box, width, height):
    """Crop is taken without resizing; translation preserves x/y and z pixel scale."""
    shift = np.array([box[0], box[1], 0], dtype=np.float32)
    if lm.face is not None:
        lm.face = lm.face + shift
    lm.hands = [h + shift for h in lm.hands]
    lm.width, lm.height = width, height
    return lm


def associate_object(detection, people, landmarks, max_hand_distance=80):
    """Only one unambiguous person with a nearby hand may own an object."""
    candidates = []
    for track in people:
        lm = landmarks.get(track.person_id)
        if lm is None or track.ambiguous or not lm.hands:
            continue
        x0, y0, x1, y1 = detection.box
        distance = min(float(np.linalg.norm(h[:, :2] - np.clip(h[:, :2], [x0,y0], [x1,y1]), axis=1).min())
                       for h in lm.hands)
        if distance <= max_hand_distance:
            candidates.append(track.person_id)
    return (candidates[0], False) if len(candidates) == 1 else (None, bool(candidates))
