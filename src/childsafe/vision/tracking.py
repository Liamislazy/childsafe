"""Conservative temporary IDs; ambiguity always discards identity history."""
from dataclasses import dataclass, replace

import numpy as np


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    area = lambda x: max(0, x[2]-x[0]) * max(0, x[3]-x[1])
    return intersection / max(area(a)+area(b)-intersection, 1e-9)


def _area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def _intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def dedupe_people(people, containment=.8, width_ratio=.75, height_ratio=.8, top_tolerance=.15):
    """YOLO sometimes adds a second 'person' box for part of the same body (typically head and
    upper body). Two overlapping boxes made the tracker mark both ambiguous and hand out two new IDs
    on every frame. A box is treated as such a duplicate only if it lies mostly inside a larger box,
    has nearly the same width, is clearly shorter, and shares the larger box's top edge (the head).
    Kept as separate people: two people crossing (same height), a child or doll on a lap (narrower)
    and a child standing in front of an adult (top edge lower); those still go through ambiguity."""
    kept = []
    for d in sorted(people, key=lambda d: _area(d.box), reverse=True):
        def duplicate_of(k):
            kh = max(k.box[3]-k.box[1], 1e-9)
            return (_intersection(d.box, k.box) >= containment * max(_area(d.box), 1e-9)
                    and (d.box[2]-d.box[0]) / max(k.box[2]-k.box[0], 1e-9) >= width_ratio
                    and (d.box[3]-d.box[1]) / kh <= height_ratio
                    and abs(d.box[1]-k.box[1]) <= top_tolerance * kh)
        dup = next((i for i, k in enumerate(kept) if duplicate_of(k)), None)
        if dup is None:
            kept.append(d)
        elif d.confidence > kept[dup].confidence:
            kept[dup] = replace(kept[dup], confidence=d.confidence)
    return kept


@dataclass
class Track:
    person_id: int
    box: tuple
    last_t: float
    ambiguous: bool = False


class PersonTracker:
    def __init__(self, max_people=4, hold_s=.8, min_iou=.25, margin=.15, primary_only=False):
        self.max_people, self.hold_s = max_people, hold_s
        # Opt-in for a one-subject setup: follow only the largest person box and ignore the rest
        # (e.g. a photo, poster or screen with a face). Do NOT use when the subject may be the
        # smaller one, such as a doll or child next to an adult.
        self.primary_only = primary_only
        self.min_iou, self.margin = min_iou, margin
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()  # monotonically increasing IDs avoid reuse across source sessions

    def update(self, detections, t):
        people = dedupe_people([d for d in detections if d.label == 'person'])
        if self.primary_only and people:
            people = [max(people, key=lambda d: _area(d.box))]
        overflow = len(people) > self.max_people
        people = sorted(people, key=lambda d: d.confidence, reverse=True)[:self.max_people]
        old = {k: v for k, v in self.tracks.items() if t-v.last_t <= self.hold_s}
        scores = {i: sorted([(iou(d.box, v.box), k) for k, v in old.items()], reverse=True)
                  for i, d in enumerate(people)}
        choices = {i: s[0][1] for i, s in scores.items() if s and s[0][0] >= self.min_iou}
        # Fast motion at low FPS (or a missed frame) can drop IoU below min_iou for the same person.
        # One person in view and one remembered ID: same person. With several people, only keep the
        # ID if the single leftover detection's centre lies inside the old box (grown by 50%).
        # Alert evidence is not carried over: the processor already resets after a >0.25 s gap.
        free_d = [i for i in range(len(people)) if i not in choices]
        free_t = [k for k in old if k not in set(choices.values())]
        if len(free_d) == 1 and len(free_t) == 1:
            ob, nb = old[free_t[0]].box, people[free_d[0]].box
            cx, cy = (nb[0]+nb[2])/2, (nb[1]+nb[3])/2
            gx, gy = (ob[2]-ob[0])*.25, (ob[3]-ob[1])*.25
            if (len(people) == 1 and len(old) == 1) or (ob[0]-gx <= cx <= ob[2]+gx and ob[1]-gy <= cy <= ob[3]+gy):
                choices[free_d[0]] = free_t[0]
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
