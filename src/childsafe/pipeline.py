"""Local segmentation, temporary IDs and per-person landmark/alert histories."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from childsafe.alerts.processor import AlertProcessor
from childsafe.vision.detection import SegmentationDetector
from childsafe.vision.tracking import PersonTracker, associate_object, to_original, iou
from childsafe.vision.elevated import ElevatedQuality, object_proximity

OBJ_COLUMNS = ['p_unsafe_raw','p_unsafe_cal','obj_class','obj_conf','p_unknown_cal']


def _same_object(prev_box, prev_label, obj):
    """Same held object as last frame? Thin objects moving toward the mouth easily drop below
    IoU 0.2 between frames, which used to reset all alert history mid-approach. Also accept the same
    label whose centre moved less than the previous box's larger side."""
    if iou(prev_box, obj.box) >= .2:
        return True
    if prev_label != obj.label:
        return False
    pc = ((prev_box[0]+prev_box[2])/2, (prev_box[1]+prev_box[3])/2)
    oc = ((obj.box[0]+obj.box[2])/2, (obj.box[1]+obj.box[3])/2)
    size = max(prev_box[2]-prev_box[0], prev_box[3]-prev_box[1])
    return math.hypot(pc[0]-oc[0], pc[1]-oc[1]) <= size


@dataclass
class SceneLandmarks:
    people: dict = field(default_factory=dict)
    boxes: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)


class FramePipeline:
    def __init__(self, cfg, ckpt, components=None, classify_all=False, classify_input='hand_crop',
                 detector=None, landmark_factory=None, classifier=None):
        self.cfg, self.components = cfg, components
        dc = cfg.get('detector', {})
        self.detector = detector if detector is not None else SegmentationDetector(
            dc.get('weights', 'models/yolo11n-seg.pt'), dc.get('min_conf', .5),
            object_confidence=dc.get('object_min_conf'), device=dc.get('device', 'auto'),
            imgsz=dc.get('imgsz', 640))
        if landmark_factory is None:
            def landmark_factory():
                from childsafe.vision.landmarks import LandmarkDetector
                lc = cfg['landmarks']
                return LandmarkDetector(cfg['paths']['face_model'], cfg['paths']['hand_model'],
                                        lc['num_hands'], lc['min_face_conf'], lc['min_hand_conf'])
        self.landmark_factory = landmark_factory
        self.classifier, self.classifier_error = classifier, ''
        if classifier is None and ckpt:
            try:
                from childsafe.vision.object_classifier import ObjectClassifier
                self.classifier = ObjectClassifier(ckpt)
            except Exception as exc:
                self.classifier_error = type(exc).__name__
        self.tracker = PersonTracker(max_people=cfg.get('runtime', {}).get('max_people', 4),
                                     hold_s=cfg.get('runtime', {}).get('identity_hold_s', 3.))
        self.states, self.session_id = {}, 0
        # Creating a MediaPipe face+hand landmarker costs ~0.4 s. Ambiguous (overlapping) people get a
        # new temporary ID on every frame, so creating one per ID stalled the loop to >1 s per frame
        # and the 0.25 s continuity rule then reset all alert history. Instances are pooled instead:
        # only people tracked in this frame hold one, and released instances are reused.
        self._landmark_pool = []

    def _new_state(self):
        from childsafe.vision.geometry import GeometryExtractor
        return dict(processor=AlertProcessor(self.cfg, self.components), geometry=GeometryExtractor(),
                    landmarks=None, landmark_failed=False, error='', features={}, object_box=None,
                    quality=ElevatedQuality(self.cfg.get('quality', {})))

    def _acquire_landmarks(self, state):
        if state['landmarks'] is not None or state['landmark_failed']:
            return
        try:
            state['landmarks'] = self._landmark_pool.pop() if self._landmark_pool else self.landmark_factory()
        except Exception as exc:
            state['error'] = type(exc).__name__
            state['landmark_failed'] = True   # do not retry a failing factory on every frame

    def _release_landmarks(self, state):
        lm, state['landmarks'] = state['landmarks'], None
        if lm is None:
            return
        if len(self._landmark_pool) <= self.tracker.max_people:
            self._landmark_pool.append(lm)
        else:
            lm.close()

    def _hand_crop_fallback(self, state, rgb, f, t):
        """YOLO (COCO classes) often misses the small object held at the mouth, and then nothing
        was classified at all. As in the original Module B design, classify the area around the hand
        instead. A confident unsafe result (>= rules.hand_crop_min_p_unsafe, stricter than the mask
        path) counts as proximity evidence measured by the hand-to-mouth distance; anything else
        stays at CAUTION through the normal rules."""
        rc = self.cfg.get('rules', {})
        if (not rc.get('hand_crop_fallback', False) or self.classifier is None
                or not f.get('mouth_visible') or not f.get('detector_ok', True)):
            return None
        from childsafe.vision.object_classifier import hand_crop_box
        h, w = rgb.shape[:2]
        box = hand_crop_box(f, w, h, self.cfg['cascade']['crop_scale'])
        if box is None:
            return None
        last = state.get('classified_hand')
        if last is not None and 0 <= t-last[0] < self.cfg.get('cascade', {}).get('classify_interval_s', .1):
            result = last[1]
        else:
            try:
                result = self.classifier(rgb, box)
                self.classifier_error = ''
            except Exception as exc:
                self.classifier_error = type(exc).__name__
                return None
            state['classified_hand'] = (t, result)
        if not result:
            return None
        threshold = self.cfg['calibration']['reject_threshold']
        confident_unsafe = (result.get('p_unsafe_cal', 0) >= rc.get('hand_crop_min_p_unsafe', .8)
                            and result.get('obj_conf', 0) >= threshold
                            and result.get('obj_class') not in ('other', 'unknown', 'empty_hand')
                            and result.get('p_unknown_cal', 0) < .5)
        f.update(object_present=True, object_label='hand crop', evidence_source='hand_crop')
        if confident_unsafe:
            f.update(proximity_evidence=True, object_d_relative=f['d_face_width'])
        return result

    def reset(self, session_id=None):
        for state in self.states.values():
            self._release_landmarks(state)
        self.states.clear()
        self.tracker.reset()
        self.session_id = self.session_id + 1 if session_id is None else session_id

    def process(self, rgb, t, *, received_t=None, frame_age_s=0., frame_valid=True, source_ok=True):
        start = time.perf_counter()
        h, w = rgb.shape[:2]
        valid = frame_valid and frame_age_s <= self.cfg.get('runtime', {}).get('stale_frame_s', .25)
        scene = SceneLandmarks()
        detections = self.detector.detect(rgb) if valid and source_ok else []
        active, retired = self.tracker.update(detections, t) if valid and source_ok else ([], set())
        active_now = {track.person_id for track in active}
        for pid, state in self.states.items():   # release first, so new IDs reuse these instances
            if pid not in active_now:
                self._release_landmarks(state)
        for track in active:
            pid = track.person_id
            if pid not in self.states:
                self.states[pid] = self._new_state()
            state = self.states[pid]
            self._acquire_landmarks(state)
            box = tuple(int(v) for v in track.box)
            box = (max(0,box[0]), max(0,box[1]), min(w,box[2]), min(h,box[3]))
            scene.boxes[pid] = box
            if state['landmarks'] is not None and box[2] > box[0] and box[3] > box[1]:
                try:
                    # Tight body crops can prevent palm detection for a gripping hand.
                    # For a lone person retain the original camera framing, then check
                    # ownership. Multiple people retain separate body crops.
                    if len(active) == 1 and not track.ambiguous:
                        crop = (0, 0, w, h)
                    else:
                        crop = box
                    lm = state['landmarks'].detect(rgb[crop[1]:crop[3],crop[0]:crop[2]], t)
                    lm = to_original(lm, crop, w, h)
                    inside = lambda points: (box[0] <= points[:,0].mean() <= box[2] and
                                             box[1] <= points[:,1].mean() <= box[3])
                    if lm.face is not None and not inside(lm.face):
                        lm.face = None
                    owned = [i for i, hand in enumerate(lm.hands) if inside(hand)]
                    lm.hands = [lm.hands[i] for i in owned]
                    if hasattr(lm, 'handedness'):
                        lm.handedness = [lm.handedness[i] for i in owned if i < len(lm.handedness)]
                    scene.people[pid] = lm
                    state['error'] = ''
                except Exception as exc:
                    state['error'] = type(exc).__name__
        ambiguous = {p.person_id for p in active if p.ambiguous}
        for pid, lm in scene.people.items():
            for hand in lm.hands:
                center = hand[:, :2].mean(0)
                for other, box in scene.boxes.items():
                    if other != pid and box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]:
                        ambiguous.update([pid, other])
        assigned = {p.person_id: [] for p in active}
        scene.objects = [d for d in detections if d.label != 'person']
        for obj in scene.objects:
            owner, unclear = associate_object(obj, active, scene.people,
                                             self.cfg.get('quality', {}).get('max_hand_object_px', 80))
            if owner is not None:
                assigned[owner].append(obj)
            elif unclear:
                ambiguous.update(assigned)
        outputs, rows = [], []
        active_ids = {p.person_id for p in active}
        for track in active:
            pid, state = track.person_id, self.states[track.person_id]
            lm = scene.people.get(pid)
            f = state['geometry'](lm) if lm is not None else dict(t=t, face_ok=0, hand_ok=0)
            f.update(schema_version=2, person_id=pid, session_id=self.session_id,
                     source_ok=source_ok, frame_valid=valid, received_t=received_t if received_t is not None else t,
                     frame_age_s=frame_age_s, detector_ok=self.detector.ok,
                     classifier_ok=self.classifier is not None and not self.classifier_error,
                     mouth_visible=lm is not None and lm.face is not None,
                     association_ambiguous=pid in ambiguous, calibrated=self.cfg.get('quality', {}).get('calibrated', False),
                     object_coverage=self.cfg.get('detector', {}).get('complete_object_coverage', False),
                     quality_reasons=[state['error']] if state['error'] else [],
                     # every non-person detection in the frame, assigned or not (diagnostics only)
                     person_x0=float(track.box[0]), person_y0=float(track.box[1]),
                     person_x1=float(track.box[2]), person_y1=float(track.box[3]),
                     yolo_people=sum(d.label == 'person' for d in detections), tracked=True,
                     scene_objects=len(scene.objects),
                     scene_object_labels=';'.join(f'{d.label}:{d.confidence:.2f}' for d in scene.objects),
                     assigned_objects=len(assigned[pid]))
            obj_result = None
            crop_box = scene.boxes[pid]
            f.update(state['quality'].measure(rgb[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]],lm,f))
            if state['error']:
                f['quality_reasons'].append('landmark model unavailable')
            objects = assigned[pid]
            obj, proximity = None, None
            if objects:
                # Select by visible mask proximity, so a long object's tip is not missed.
                ranked = [(object_proximity(d,lm,f,self.cfg.get('quality',{})),d) for d in objects]
                proximity, obj = min(ranked,key=lambda item: item[0]['object_d_relative']
                             if math.isfinite(item[0]['object_d_relative']) else
                             item[0]['candidate_object_d_relative']
                             if math.isfinite(item[0]['candidate_object_d_relative']) else math.inf)
                if state['object_box'] is not None and not _same_object(state['object_box'], state.get('object_label'), obj):
                    state['processor'].reset()
                    f['history_reset'] = True
                state['object_box'], state['object_label'] = obj.box, obj.label
                last = state.get('classified')
                recent = (last is not None and not f.get('history_reset') and
                          state['processor'].obj is not None and
                          0 <= t-last[0] < self.cfg.get('cascade', {}).get('classify_interval_s', .1) and
                          iou(last[1], obj.box) >= .5)
                # Same object classified moments ago: the processor keeps that result (valid 1 s).
                if self.classifier is not None and pid not in ambiguous and not recent:
                    try:
                        box = tuple(int(v) for v in obj.box)
                        box = (max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3]))
                        if box[2]-box[0] > 8 and box[3]-box[1] > 8:
                            obj_result = self.classifier(rgb, box)
                            state['classified'] = (t, obj.box)
                        self.classifier_error = ''
                    except Exception as exc:
                        self.classifier_error = type(exc).__name__
                        f['classifier_ok'] = False
            else:
                state['object_box'] = None
                state['processor'].obj = None
            if pid in ambiguous:
                f['history_reset'] = True
                state['processor'].reset()
                state['geometry'].__init__()
            f.update(proximity if proximity is not None else object_proximity(obj,lm,f,self.cfg.get('quality',{})))
            f['object_present'] = obj is not None
            f.setdefault('evidence_source', 'mask' if f.get('proximity_evidence') else '')
            hand_distance = f.get('d_face_width',math.nan)
            suspicious_hand = math.isfinite(hand_distance) and hand_distance <= self.cfg['temporal']['d_exit']
            if obj is None and suspicious_hand and pid not in ambiguous:
                hand_result = self._hand_crop_fallback(state, rgb, f, t)
                if hand_result is not None:
                    obj_result = hand_result
            if not f['proximity_evidence'] and (obj is not None or suspicious_hand):
                f['quality_reasons'].append('no reliable held-object mask evidence')
            out = state['processor'].step(f, obj_result)
            out.update(person_id=pid, box=track.box, session_id=self.session_id, visible=True)
            state['box'] = track.box
            f.update({k: obj_result.get(k, math.nan) if obj_result else ('' if k == 'obj_class' else math.nan)
                      for k in OBJ_COLUMNS})
            state['features'] = f
            rows.append(f)
            outputs.append(out)
        for pid in list(self.states):
            if pid in active_ids:
                continue
            state = self.states[pid]
            f = dict(state['features'])
            f.update(t=t, schema_version=2, person_id=pid, session_id=self.session_id,
                     face_ok=0, mouth_visible=False, frame_valid=valid, source_ok=source_ok,
                     person_left=pid in retired, received_t=received_t if received_t is not None else t,
                     frame_age_s=frame_age_s)
            f['object_present'] = False
            f['tracked'] = False   # held ID, not detected in this frame
            f.update(proximity_evidence=False,mask_reliable=False,object_d_relative=math.nan,
                     object_d_px=math.nan,detector_ok=self.detector.ok,
                     classifier_ok=self.classifier is not None and not self.classifier_error)
            for column in OBJ_COLUMNS:
                f[column] = '' if column == 'obj_class' else math.nan
            out = state['processor'].step(f)
            # held ID: not detected in this frame (kept so a brief miss does not reset its history)
            out.update(person_id=pid, session_id=self.session_id, visible=False, box=state.get('box'))
            rows.append(f)
            if pid not in retired:
                outputs.append(out)
            if pid in retired or not source_ok:
                self._release_landmarks(state)
                del self.states[pid]
        inference_s = time.perf_counter()-start
        for f in rows:
            f['inference_s'] = inference_s
        level = max((o['level'] for o in outputs), default=0)
        quality = 'unobservable' if not outputs else (
            'unobservable' if any(o['observation'] == 'unobservable' for o in outputs) else
            'degraded' if any(o['observation'] == 'degraded' for o in outputs) else 'observable')
        aggregate = dict(t=t, level=level, reason='; '.join(sorted({o['reason'] for o in outputs if o['reason']})),
                         observation=quality, green=bool(outputs) and all(o['green'] for o in outputs),
                         persons=outputs, feature_rows=rows, inference_s=inference_s,
                         calibrated=self.cfg.get('quality',{}).get('calibrated',False),
                         observation_reasons=sorted({r for o in outputs for r in o['observation_reasons']}),
                         d=math.nan, near=False, dwell=0., obj_status='none', obj_class='', p_unsafe=math.nan)
        if not outputs:
            aggregate['reason'] = 'no observable person; detector limited' if not self.detector.ok else 'no observable person'
            # Preserve empty/failed source intervals in replay and coverage denominators.
            f = dict(t=t,schema_version=2,person_id=0,session_id=self.session_id,source_ok=source_ok,
                     frame_valid=valid,mouth_visible=False,face_ok=0,hand_ok=0,
                     detector_ok=self.detector.ok,classifier_ok=self.classifier is not None,
                     frame_age_s=frame_age_s,inference_s=inference_s,
                     received_t=received_t if received_t is not None else t)
            rows.append(f)
        return scene, rows[0] if rows else {}, aggregate

    def close(self):
        for state in self.states.values():
            if state['landmarks'] is not None:
                state['landmarks'].close()
                state['landmarks'] = None
        for lm in self._landmark_pool:
            lm.close()
        self._landmark_pool.clear()
