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
            dc.get('weights', 'models/yolo11n-seg.pt'), dc.get('min_conf', .5))
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
        self.tracker = PersonTracker(max_people=cfg.get('runtime', {}).get('max_people', 4))
        self.states, self.session_id = {}, 0

    def _new_state(self):
        from childsafe.vision.geometry import GeometryExtractor
        state = dict(processor=AlertProcessor(self.cfg, self.components), geometry=GeometryExtractor(),
                     landmarks=None, error='', features={}, object_box=None,
                     quality=ElevatedQuality(self.cfg.get('quality', {})))
        try:
            state['landmarks'] = self.landmark_factory()
        except Exception as exc:
            state['error'] = type(exc).__name__
        return state

    def reset(self, session_id=None):
        self.close()
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
        for track in active:
            pid = track.person_id
            if pid not in self.states:
                self.states[pid] = self._new_state()
            state = self.states[pid]
            box = tuple(int(v) for v in track.box)
            box = (max(0,box[0]), max(0,box[1]), min(w,box[2]), min(h,box[3]))
            scene.boxes[pid] = box
            if state['landmarks'] is not None and box[2] > box[0] and box[3] > box[1]:
                try:
                    lm = state['landmarks'].detect(rgb[box[1]:box[3],box[0]:box[2]], t)
                    scene.people[pid] = to_original(lm, box, w, h)
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
                     quality_reasons=[state['error']] if state['error'] else [])
            obj_result = None
            crop_box = scene.boxes[pid]
            f.update(state['quality'].measure(rgb[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]],lm,f))
            if state['error']:
                f['quality_reasons'].append('landmark model unavailable')
            objects = assigned[pid]
            obj = None
            if objects:
                # Select by visible mask proximity, so a long object's tip is not missed.
                ranked = [(object_proximity(d,lm,f,self.cfg.get('quality',{})),d) for d in objects]
                _, obj = min(ranked,key=lambda item: item[0]['object_d_relative']
                             if math.isfinite(item[0]['object_d_relative']) else math.inf)
                if state['object_box'] is not None and iou(state['object_box'], obj.box) < .2:
                    state['processor'].reset()
                    f['history_reset'] = True
                state['object_box'] = obj.box
                if self.classifier is not None and pid not in ambiguous:
                    try:
                        box = tuple(int(v) for v in obj.box)
                        box = (max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3]))
                        if box[2]-box[0] > 8 and box[3]-box[1] > 8:
                            obj_result = self.classifier(rgb, box)
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
            f.update(object_proximity(obj,lm,f,self.cfg.get('quality',{})))
            f['object_present'] = obj is not None
            hand_distance = f.get('d_face_width',math.nan)
            suspicious_hand = math.isfinite(hand_distance) and hand_distance <= self.cfg['temporal']['d_exit']
            if not f['proximity_evidence'] and (obj is not None or suspicious_hand):
                f['quality_reasons'].append('no reliable held-object mask evidence')
            out = state['processor'].step(f, obj_result)
            out.update(person_id=pid, box=track.box, session_id=self.session_id)
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
            f.update(proximity_evidence=False,mask_reliable=False,object_d_relative=math.nan,
                     object_d_px=math.nan,detector_ok=self.detector.ok,
                     classifier_ok=self.classifier is not None and not self.classifier_error)
            for column in OBJ_COLUMNS:
                f[column] = '' if column == 'obj_class' else math.nan
            out = state['processor'].step(f)
            out.update(person_id=pid, session_id=self.session_id)
            rows.append(f); outputs.append(out)
            if pid in retired or not source_ok:
                if state['landmarks'] is not None:
                    state['landmarks'].close()
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
