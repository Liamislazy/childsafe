"""Local segmentation, temporary IDs and per-person landmark/alert histories."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time

import cv2
import numpy as np

from childsafe.alerts.processor import AlertProcessor
from childsafe.vision.detection import SegmentationDetector
from childsafe.vision.tracking import PersonTracker, associate_object, iou, resolve_hands
from childsafe.vision.elevated import ElevatedQuality, object_proximity
from childsafe.vision.objects import MaskEvidence, assign_objects, same_object as _same_object, sharp_without_hand
from childsafe.vision.people import detect_person
from childsafe.runtime.contracts import PersonState

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
        from childsafe.evaluation.sessions import config_hash,file_hash,code_hash
        from pathlib import Path
        from childsafe.vision.detection import select_weights
        weights = select_weights(cfg.get('detector', {}))
        self.provenance = dict(config_sha256=config_hash(cfg),code_sha256=code_hash(),
                               classifier_sha256=file_hash(ckpt) if ckpt and Path(ckpt).is_file() else None,
                               detector_sha256=file_hash(weights) if weights and Path(weights).is_file() else None)
        dc = cfg.get('detector', {})
        self.detector = detector if detector is not None else SegmentationDetector(
            weights, dc.get('min_conf', .5),
            object_confidence=dc.get('object_min_conf'), device=dc.get('device', 'auto'),
            imgsz=dc.get('imgsz', 640), cpu_threads=cfg.get('runtime', {}).get('cpu_threads', 0))
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
                                     hold_s=cfg.get('runtime', {}).get('identity_hold_s', 3.),
                                     primary_only=cfg.get('runtime', {}).get('primary_person_only', False))
        self.states: dict[int,PersonState] = {}
        self.session_id = 0
        # Creating a MediaPipe face+hand landmarker costs ~0.4 s. Ambiguous (overlapping) people get a
        # new temporary ID on every frame, so creating one per ID stalled the loop to >1 s per frame
        # and the 0.25 s continuity rule then reset all alert history. Instances are pooled instead:
        # only people tracked in this frame hold one, and released instances are reused.
        self._landmark_pool = []

    def _new_state(self) -> PersonState:
        from childsafe.vision.geometry import GeometryExtractor
        return dict(processor=AlertProcessor(self.cfg, self.components), geometry=GeometryExtractor(),
                    landmarks=None, landmark_failed=False, error='', classifier_error='', features={}, object_box=None,
                    quality=ElevatedQuality(self.cfg.get('quality', {})),
                    mask_evidence=MaskEvidence(self.cfg.get('quality',{}),self.cfg['temporal']['max_gap_s']))

    @staticmethod
    def _clear_classification(state):
        for cache in ('classified', 'classified_hand', 'obj_result'):
            state.pop(cache, None)
        state['processor'].obj = None

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
        if len(self._landmark_pool) < self.tracker.max_people:
            self._landmark_pool.append(lm)
        else:
            lm.close()

    def _zoom_detect(self, rgb, scene, active, ambiguous, deadline=None):
        """Second detector pass on a crop around each person's hands and mouth. The full frame is
        resized to imgsz (640), so a small or distant held object is only a few pixels wide; in the
        crop it is several times larger. Objects found there are mapped back to full-frame
        coordinates (mask included) and merged with the full-frame detections (higher score wins
        when they overlap). Only for the real detector (fakes in tests do not declare support)."""
        dc = self.cfg.get('detector', {})
        if not dc.get('roi_zoom', False) or not getattr(self.detector, 'supports_roi', False):
            return 0
        h, w = rgb.shape[:2]
        found = 0
        budget=dc.get('roi_budget_per_frame',self.tracker.max_people)
        if not isinstance(budget,int) or isinstance(budget,bool) or budget<0:
            raise ValueError('roi_budget_per_frame must be a nonnegative integer')
        cursor=getattr(self,'_roi_cursor',0)%max(len(active),1)
        ordered=active[cursor:]+active[:cursor]
        self._roi_cursor=(cursor+1)%max(len(active),1)
        for track in ordered:
            if self.roi_passes>=budget or (deadline is not None and time.perf_counter()>=deadline):
                break
            lm = scene.people.get(track.person_id)
            if track.person_id in ambiguous or lm is None or not lm.hands:
                continue
            # A full-frame mask already resolves the held object: do not repeat
            # segmentation just to obtain a second view of the same visible pixels.
            if dc.get('roi_skip_resolved', True):
                resolved = False
                for detection in scene.objects:
                    if detection.mask is None or detection.confidence < self.cfg.get('quality', {}).get('min_mask_conf', .6):
                        continue
                    owner, unclear = associate_object(detection, active, scene.people,
                        max_distance_ratio=self.cfg.get('quality',{}).get('max_hand_object_ratio',1.5))
                    if owner == track.person_id and not unclear:
                        if lm.face is None:
                            continue
                        from childsafe.vision.geometry import face_measures
                        features = dict(face_width=face_measures(lm.face)['face_width'], hand_ok=True,
                                        mouth_visible=True, association_ambiguous=False, quality_reasons=[])
                        proximity = object_proximity(detection, lm, features, self.cfg.get('quality', {}))
                        if proximity['proximity_evidence'] and proximity['object_d_relative'] < self.cfg['temporal']['d_exit']:
                            resolved = True
                            break
                if resolved:
                    continue
            pts = [hd[:, :2] for hd in lm.hands]
            if lm.face is not None:
                pts.append(lm.face[[13, 14, 78, 308], :2])
            pts = np.concatenate(pts)
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            side = max(x1 - x0, y1 - y0) * dc.get('roi_scale', 1.8)
            side = min(max(side, dc.get('roi_min_px', 192)), min(h, w))
            if side * side > dc.get('roi_max_frame_frac', .5) * h * w:
                continue   # crop would be most of the frame: nothing to gain
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bx0 = int(min(max(cx - side / 2, 0), w - side)); by0 = int(min(max(cy - side / 2, 0), h - side))
            bx1, by1 = int(bx0 + side), int(by0 + side)
            self.roi_passes+=1
            for d in self._timed('detector_s',self.detector.detect,np.ascontiguousarray(rgb[by0:by1, bx0:bx1])):
                if d.label == 'person':
                    continue
                box = (d.box[0] + bx0, d.box[1] + by0, d.box[2] + bx0, d.box[3] + by0)
                mask = None
                if d.mask is not None:
                    mask = np.zeros((h, w), bool)
                    mask[by0:by0 + d.mask.shape[0], bx0:bx0 + d.mask.shape[1]] = d.mask
                obj = replace(d, box=box, mask=mask)
                same = [k for k, o in enumerate(scene.objects) if iou(o.box, box) > .5]
                if same and all(scene.objects[k].confidence >= obj.confidence for k in same):
                    continue
                scene.objects = [o for k, o in enumerate(scene.objects) if k not in same] + [obj]
                found += 1
        return found

    def _confident_unsafe(self, result):
        rc = self.cfg.get('rules', {})
        return (result.get('p_unsafe_cal', 0) >= rc.get('hand_crop_min_p_unsafe', .8)
                and result.get('obj_conf', 0) >= self.cfg['calibration']['reject_threshold']
                and result.get('obj_class') not in ('other', 'unknown', 'empty_hand')
                and result.get('p_unknown_cal', 0) < .5)

    def _hand_crop_fallback(self, state, rgb, f, t):
        """YOLO (COCO classes) often misses the small object held at the mouth, and then nothing
        was classified at all. As in the original Module B design, classify the area around the hand
        instead. Classification describes the crop; without a localized reliable
        object mask this is suspicion only, never confirmed proximity evidence."""
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
                result = self._timed('classifier_s',self.classifier,rgb,box)
                self.classifier_error = ''
                state['classifier_error'] = ''
            except Exception as exc:
                self.classifier_error = type(exc).__name__
                state['classifier_error'] = self.classifier_error
                self._clear_classification(state)
                return None
            state['classified_hand'] = (t, result)
        if not result:
            return None
        f.update(evidence_source='hand_crop',classification_source='hand_crop',distance_source='hand')
        return result

    def reset(self, session_id=None):
        for state in self.states.values():
            self._release_landmarks(state)
        self.states.clear()
        self.tracker.reset()
        self.session_id = self.session_id + 1 if session_id is None else session_id

    def _timed(self, stage, function, *args, **kwargs):
        if stage=='detector_s':
            self.detector_calls=getattr(self,'detector_calls',0)+1
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            if not hasattr(self,'timings'):
                self.timings = {}
            self.timings[stage] = self.timings.get(stage,0.) + time.perf_counter()-start

    def process(self, rgb, t, *, received_t=None, frame_age_s=0., frame_valid=True, source_ok=True,
                processing_deadline=None):
        start = time.perf_counter()
        self.timings = {}
        self.detector_calls,self.roi_passes=0,0
        import copy
        prior_processors = ({pid:copy.deepcopy(state['processor']) for pid,state in self.states.items()}
                            if processing_deadline is not None else {})
        h, w = rgb.shape[:2]
        valid = frame_valid and frame_age_s <= self.cfg.get('runtime', {}).get('stale_frame_s', .25)
        valid = valid and (processing_deadline is None or start <= processing_deadline)
        scene = SceneLandmarks()
        detections = self._timed('detector_s',self.detector.detect,rgb) if valid and source_ok else []
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
            if track.history_reset:
                state['processor'].reset()
                self._clear_classification(state)
                state['geometry'].__init__()
                state['quality'] = ElevatedQuality(self.cfg.get('quality', {}))
                state['mask_evidence'].reset()
                state['object_box'] = None
            if processing_deadline is None or time.perf_counter()<processing_deadline:
                self._acquire_landmarks(state)
            box = tuple(int(v) for v in track.box)
            box = (max(0,box[0]), max(0,box[1]), min(w,box[2]), min(h,box[3]))
            scene.boxes[pid] = box
            if (state['landmarks'] is not None and box[2] > box[0] and box[3] > box[1]
                    and (processing_deadline is None or time.perf_counter()<processing_deadline)):
                try:
                    lm = self._timed('landmarks_s',detect_person,state['landmarks'],rgb,box,t,
                                     self.cfg.get('landmarks',{}),len(active)==1 and not track.ambiguous)
                    scene.people[pid] = lm
                    state['error'] = ''
                except Exception as exc:
                    state['error'] = type(exc).__name__
        ambiguous = {p.person_id for p in active if p.ambiguous}
        # Whose hand is whose: duplicates across crops go to the person whose mouth they are
        # clearly closer to; only truly unclear hands make people ambiguous (see resolve_hands).
        ambiguous |= resolve_hands(scene.people, scene.boxes)
        scene.objects = [d for d in detections if d.label != 'person']
        roi_found = self._zoom_detect(rgb, scene, active, ambiguous,processing_deadline) if valid and source_ok else 0
        assignments=assign_objects(scene.objects,active,scene.people,
                                   self.cfg.get('quality',{}).get('max_hand_object_ratio',1.5))
        assigned=assignments.owned
        ambiguous.update(assignments.ambiguous)
        # No hand found (a fist hides it): a confident knife/scissors mostly inside one person's box
        # is still that person's held object, for classification and the held-sharp rule.
        rules = self.cfg.get('rules', {})
        claimed = {id(d) for objs in assigned.values() for d in objs}
        for pid, objs in sharp_without_hand([d for d in scene.objects if id(d) not in claimed], active,
                                            scene.people, set(rules.get('held_sharp_labels', [])),
                                            rules.get('held_sharp_min_conf', .6)).items():
            if pid not in ambiguous:
                assigned[pid].extend(objs)
        outputs, rows = [], []
        active_ids = {p.person_id for p in active}
        for track in active:
            pid, state = track.person_id, self.states[track.person_id]
            lm = scene.people.get(pid)
            f = state['geometry'](lm) if lm is not None else dict(t=t, face_ok=0, hand_ok=0)
            f.update(schema_version=2, person_id=pid, session_id=self.session_id,
                     history_reset=track.history_reset, person_reset=track.history_reset,
                     source_ok=source_ok, frame_valid=valid, received_t=received_t if received_t is not None else t,
                     frame_age_s=frame_age_s, detector_ok=self.detector.ok,
                     classifier_ok=self.classifier is not None and not state['classifier_error'],
                     mouth_visible=lm is not None and lm.face is not None,
                     association_ambiguous=pid in ambiguous, calibrated=self.cfg.get('quality', {}).get('calibrated', False),
                     object_coverage=self.cfg.get('detector', {}).get('complete_object_coverage', False),
                     quality_reasons=[state['error']] if state['error'] else [],
                     # every non-person detection in the frame, assigned or not (diagnostics only)
                     person_x0=float(track.box[0]), person_y0=float(track.box[1]),
                     person_x1=float(track.box[2]), person_y1=float(track.box[3]),
                     yolo_people=sum(d.label == 'person' for d in detections), tracked=True,
                     scene_objects=len(scene.objects), roi_objects=roi_found,
                     scene_object_labels=';'.join(f'{d.label}:{d.confidence:.2f}' for d in scene.objects),
                     assigned_objects=len(assigned[pid]))
            obj_result = None
            # Landmarks are in original coordinates; quality crops must use that same image.
            f.update(self._timed('quality_s',state['quality'].measure,rgb,lm,f))
            if state['error']:
                f['quality_reasons'].append('landmark model unavailable')
            objects = assigned[pid]
            # Knife/scissors linked to this person's hand (rules.held_sharp_*): the processor raises
            # DANGER for a confident one wherever the hand is. Ambiguous ownership never counts.
            rules = self.cfg.get('rules', {})
            sharp = max((d for d in objects if d.label in rules.get('held_sharp_labels', [])),
                        key=lambda d: d.confidence, default=None)
            if sharp is None or pid in ambiguous:
                f.update(held_sharp_label='', held_sharp_conf=math.nan, held_sharp_confident=False)
            else:
                f.update(held_sharp_label=sharp.label, held_sharp_conf=sharp.confidence,
                         held_sharp_confident=sharp.confidence >= rules.get('held_sharp_min_conf', .6))
            obj, proximity = None, None
            if objects:
                # Select by visible mask proximity, so a long object's tip is not missed.
                ranked = [(object_proximity(d,lm,f,self.cfg.get('quality',{})),d) for d in objects]
                proximity, obj = min(ranked,key=lambda item: item[0]['object_d_relative']
                             if math.isfinite(item[0]['object_d_relative']) else
                             item[0]['candidate_object_d_relative']
                             if math.isfinite(item[0]['candidate_object_d_relative']) else math.inf)
                if state['object_box'] is not None and not _same_object(state['object_box'], state.get('object_label'), obj):
                    state['processor'].reset(keep_held_sharp=True)
                    self._clear_classification(state)
                    f['history_reset'] = f['object_switched'] = True
                    state['mask_evidence'].reset()
                state['object_box'], state['object_label'] = obj.box, obj.label
                last = state.get('classified')
                recent = (last is not None and not f.get('history_reset') and
                          state['processor'].obj is not None and
                          0 <= t-last[0] < self.cfg.get('cascade', {}).get('classify_interval_s', .1) and
                          iou(last[1], obj.box) >= .5)
                # Same object classified moments ago: the processor keeps that result (valid 1 s).
                if (self.classifier is not None and pid not in ambiguous and not recent
                        and (processing_deadline is None or time.perf_counter()<processing_deadline)):
                    try:
                        box = tuple(int(v) for v in obj.box)
                        box = (max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3]))
                        if box[2]-box[0] > 8 and box[3]-box[1] > 8:
                            obj_result = self._timed('classifier_s',self.classifier,rgb,box)
                            state['classified'] = (t, obj.box)
                        else:
                            self._clear_classification(state)
                            f['quality_reasons'].append('object crop too small for classification')
                        self.classifier_error = ''
                        state['classifier_error'] = ''
                    except Exception as exc:
                        self.classifier_error = type(exc).__name__
                        state['classifier_error'] = self.classifier_error
                        self._clear_classification(state)
                        f['classifier_ok'] = False
            else:
                state['object_box'] = None
                state['processor'].obj = None
                state.pop('classified', None)
                state.pop('obj_result', None)
            if pid in ambiguous:
                f['history_reset'] = True
                state['processor'].reset()
                self._clear_classification(state)
                state['geometry'].__init__()
                state['mask_evidence'].reset()
            raw_proximity=proximity if proximity is not None else object_proximity(obj,lm,f,self.cfg.get('quality',{}))
            f.update(state['mask_evidence'].apply(obj,t,raw_proximity))
            if f['mask_reliable'] and not f['mask_temporally_stable']:
                f['quality_reasons'].append('object mask not temporally stable')
            f['object_present'] = obj is not None
            f.setdefault('evidence_source', 'mask' if f.get('proximity_evidence') else '')
            d_object = (f['object_d_relative'] if f.get('proximity_evidence')
                        else f.get('candidate_object_d_relative', math.nan))
            hand_distance = f.get('d_face_width',math.nan)
            d_exit = self.cfg['temporal']['d_exit']
            suspicious_hand = math.isfinite(hand_distance) and hand_distance <= d_exit
            if (obj is None and suspicious_hand and pid not in ambiguous
                    and (processing_deadline is None or time.perf_counter()<processing_deadline)):
                hand_result = self._hand_crop_fallback(state, rgb, f, t)
                if hand_result is not None:
                    obj_result = hand_result
            # Keep both distances for diagnostics/suspicion. Only a reliable visible
            # object mask can establish proximity evidence; classification is contextual.
            if obj is not None and obj_result is not None:
                state['obj_result'] = (t, obj_result)
            # Classification cannot upgrade a weak mask/box or hand position to contact evidence.
            f['d_object'], f['d_hand'] = d_object, hand_distance
            f['distance_source'] = ('object' if math.isfinite(d_object) and
                                    (not math.isfinite(hand_distance) or d_object <= hand_distance) else 'hand')
            if not f['proximity_evidence'] and (obj is not None or suspicious_hand):
                f['quality_reasons'].append('no reliable held-object mask evidence')
            f['classifier_ok'] = self.classifier is not None and not state['classifier_error']
            if processing_deadline is not None and time.perf_counter() > processing_deadline:
                f.update(frame_valid=False, mouth_visible=False, proximity_evidence=False)
                f['quality_reasons'].append('frame expired during inference')
                obj_result = None
                self._clear_classification(state)
                state['mask_evidence'].reset()
            out = state['processor'].step(f, obj_result)
            f.update({k:out[k] for k in ('hazard_status','hazard_type','hazard_policy_version')})
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
                     classifier_ok=self.classifier is not None and not state['classifier_error'])
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
        from childsafe.runtime.resources import peak_memory_bytes
        memory = peak_memory_bytes()
        if self.classifier is not None:
            self.classifier_error = next((state['classifier_error'] for state in self.states.values()
                                          if state['classifier_error']), '')
        for f in rows:
            f['inference_s'] = inference_s
            f.update({name:self.timings.get(name,0.) for name in ('detector_s','landmarks_s','classifier_s','quality_s')})
            f['other_s'] = max(0.,inference_s-sum(self.timings.values()))
            f['peak_process_memory_bytes'] = memory
        # A later person's work may expire the whole shared source sample.
        if processing_deadline is not None and time.perf_counter() > processing_deadline:
            outputs = []
            for f in rows:
                f.update(frame_valid=False,mouth_visible=False,face_ok=False,hand_ok=False,
                         proximity_evidence=False,history_reset=False,d_last_mouth=math.nan)
                f.setdefault('quality_reasons',[]).append('frame expired during inference')
                state = self.states.get(f['person_id'])
                if state is not None:
                    if f['person_id'] in prior_processors:
                        state['processor'] = prior_processors[f['person_id']]
                    else:
                        state['processor'].reset()
                    self._clear_classification(state)
                    state['mask_evidence'].reset()
                    result = state['processor'].step(f)
                    result.update(person_id=f['person_id'],session_id=self.session_id,
                                  visible=False,box=state.get('box'))
                    outputs.append(result)
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
            f.update({name:self.timings.get(name,0.) for name in ('detector_s','landmarks_s','classifier_s','quality_s')})
            f['other_s'] = max(0.,inference_s-sum(self.timings.values()))
            f['peak_process_memory_bytes'] = memory
        for f in rows:
            f.update(self.provenance)
            f.update(detector_calls=self.detector_calls,roi_detector_calls=self.roi_passes)
        return scene, rows[0] if rows else {}, aggregate

    def close(self):
        for state in self.states.values():
            if state['landmarks'] is not None:
                state['landmarks'].close()
                state['landmarks'] = None
        for lm in self._landmark_pool:
            lm.close()
        self._landmark_pool.clear()
