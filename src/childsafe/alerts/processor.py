"""Per-frame alert processor: smoothing (B2), lost-landmark handling (B2), approach speed (B3-lite),
temporal rule (D1), cascade gating (D2) and the alert matrix (D3).

The same class runs live (scripts/run_webcam.py) and offline on logged features
(scripts/replay.py), so every threshold and every ablation (E4) can be evaluated on
recorded feature logs without touching video again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from childsafe.alerts.temporal import Hysteresis, TimeWindow
from childsafe.alerts.observation import observation, image_observation
from childsafe.alerts.hazards import assess_hazard, detector_hazard, strong_detector_category
from childsafe.vision.smoothing import EMA, NoFilter, make_filter

NORMAL, CAUTION, DANGER = 0, 1, 2
LEVEL_NAMES = {NORMAL: "normal", CAUTION: "caution", DANGER: "danger"}


@dataclass
class Components:
    """E4 ablation switches. All True = full system."""
    smoothing: bool = True
    hysteresis: bool = True
    n_of_m: bool = True
    cascade: bool = True
    object_gate: bool = True     # False: ignore Module A, every near event is treated as unsafe
    calibration: bool = True     # False: use uncalibrated P(unsafe)
    lost_landmark: bool = True
    approach: bool = True
    cooldown: bool = True

    @classmethod
    def from_off(cls, off: list[str] | None):
        c = cls()
        for name in off or []:
            if name not in {f.name for f in fields(cls)}:
                raise ValueError(f"unknown component {name}")
            setattr(c, name, False)
        return c


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class AlertProcessor:
    def __init__(self, cfg: dict, components: Components | None = None):
        self.cfg = cfg
        self.c = components or Components()
        self.norm = cfg["landmarks"]["normalizer"]
        tc, sc = cfg["temporal"], cfg["smoothing"]
        self.filter = make_filter(sc) if self.c.smoothing else NoFilter()
        self.speed_filter = EMA(0.3)
        enter = tc["d_enter"]
        self.hyst = Hysteresis(enter, tc["d_exit"] if self.c.hysteresis else enter)
        self.vote = TimeWindow(tc.get('window_s',10/30), tc.get('required_evidence_s',3/30)
                               if self.c.n_of_m else 0.,tc.get('max_gap_s',.25))
        caution_s = cfg['rules'].get('caution_evidence_s', .15)
        self.candidate_vote = TimeWindow(max(tc.get('window_s',10/30), caution_s),
                                         caution_s,tc.get('max_gap_s',.25))
        self.reset()

    def reset(self):
        self.filter.reset(); self.speed_filter.reset(); self.hyst.reset(); self.vote.reset()
        self.candidate_vote.reset()
        self.frame = 0
        self.prev_d = self.prev_t = None
        self.last_face_t = self.last_d = None
        self.near_since = None
        self.obj = None            # (t, p_unsafe_raw, p_unsafe_cal, class)
        self.danger_on = False
        self.danger_block_until = -math.inf
        self.last_evidence_t = None
        self.last_level = NORMAL
        self.last_alert_t = None
        self.event_active = False
        self.event_reliable = False
        self.last_step_t = None
        self.confirmed_dwell = 0.
        self.prev_confirmed = False
        self.last_confirmed_t = None
        self.paused_dwell = 0.
        self.last_classify_t = None

    # ---- D2 cascade -------------------------------------------------------------------
    def wants_classification(self, f: dict) -> bool:
        """Call before running Module A on this frame."""
        if not f.get("hand_ok"):
            return False
        if not self.c.cascade:
            return True
        cc = self.cfg["cascade"]
        d = f.get(f"d_{self.norm}")
        if _isnan(d):
            d = f.get("d_last_mouth")   # face hidden: judge by last known mouth position
        return (not _isnan(d) and d < cc['d_far'] and
                (self.last_classify_t is None or f['t']-self.last_classify_t >= cc.get('classify_interval_s',.1)-1e-9))

    # ---- main step --------------------------------------------------------------------
    def step(self, f: dict, obj: dict | None = None) -> dict:
        """f: feature dict from GeometryExtractor; obj: ObjectClassifier output for this frame, if run."""
        if f.get('history_reset', False):
            self.reset()
        t = f["t"]
        tc, sc, rc = self.cfg["temporal"], self.cfg["smoothing"], self.cfg["rules"]
        self.frame += 1
        dt = t-self.last_step_t if self.last_step_t is not None else 0.
        gap_ok = 0 < dt <= tc.get('max_gap_s',.25)
        if self.last_step_t is not None and not gap_ok:
            self.filter.reset(); self.speed_filter.reset(); self.hyst.reset(); self.vote.reset()
            self.candidate_vote.reset()
            self.prev_d = self.prev_t = None
            self.confirmed_dwell = 0.
            self.prev_confirmed = False
            self.last_confirmed_t, self.paused_dwell = None, 0.
            self.obj = None
        self.last_step_t = t
        quality, quality_reasons = observation(f)
        image_quality, image_reasons = image_observation(f)
        modern = 'schema_version' in f
        valid = f.get('frame_valid', True) and f.get('source_ok', True)
        visible = bool(f.get('mouth_visible', f.get('face_ok'))) and valid
        if modern and (not valid or f.get('association_ambiguous')):
            obj = None
            if not valid:
                self.obj = None
        if modern and (not f.get('classifier_ok', False) or not f.get('detector_ok', False)):
            obj = None
            self.obj = None
        if modern and f.get('object_present') is False and f.get('evidence_source')!='hand_crop':
            obj = None
            self.obj = None
        if obj is not None and not _isnan(obj.get("p_unsafe_cal")):
            self.last_classify_t = t
            self.obj = (t, obj["p_unsafe_raw"], obj["p_unsafe_cal"], obj.get("obj_class", ""),
                        obj.get('obj_conf',math.nan),obj.get('p_unknown_cal',0.))

        # A bare hand near the face is common (scratching, resting the chin).
        # Only a localized object's distance may start a modern risk alert.
        # Keep hand-crop classification as diagnostics, not proof of an object.
        candidate_d = f.get('candidate_object_d_relative', math.nan)
        object_present = f.get('object_present', bool(f.get('proximity_evidence',False)
                                                     or not _isnan(candidate_d)))
        d_raw = f.get(f"d_{self.norm}")
        if modern:
            d_raw = math.nan
            if object_present and f.get('hand_ok',False) and f.get('detector_ok',False):
                d_raw = f.get('object_d_relative') if f.get('proximity_evidence',False) else candidate_d
        if not visible or (modern and f.get('association_ambiguous')):
            d_raw = math.nan
        if visible and not _isnan(d_raw) and (not modern or f.get('proximity_evidence',False)):
            self.last_evidence_t = t
        lost_near = False
        if _isnan(d_raw):
            d = None
            if self.c.lost_landmark and self.last_face_t is not None and not f.get("face_ok"):
                gap = t - self.last_face_t
                was_near = self.last_d is not None and self.last_d < tc["d_exit"]
                if was_near and gap <= sc["lost_hold_s"]:
                    lost_near = True
            if d is None:
                self.filter.reset(); self.speed_filter.reset()
                self.prev_d = None
        else:
            d = self.filter(float(d_raw), t)
            if f.get("face_ok"):
                self.last_face_t, self.last_d = t, d

        # approach speed and time-to-contact on the smoothed distance
        speed, ttc = 0.0, math.inf
        if d is not None and self.prev_d is not None and t > self.prev_t:
            speed = self.speed_filter(-(d - self.prev_d) / (t - self.prev_t), t)  # >0 = approaching
            if speed > 1e-3:
                ttc = d / speed
        self.prev_d, self.prev_t = (d, t) if d is not None else (None, None)

        near_frame = self.hyst(d if d is not None else math.nan)
        near = self.vote(t,near_frame and (not modern or f.get('proximity_evidence',False)))
        if near and self.near_since is None:
            self.near_since = t
        elif not near:
            self.near_since = None
        confirmed = near and visible and (not modern or f.get('proximity_evidence',False))
        # Dwell only grows over observed confirmed-to-confirmed intervals (never from held coordinates).
        # A brief detector dropout (thin objects flicker for a frame) pauses the dwell instead of
        # restarting it, so a sustained approach can still reach dwell_danger_s.
        # Shown dwell is 0 while evidence is missing; a dwell paused by a dropout resumes only if
        # confirmation returns within dropout_grace_s of the last confirmed frame.
        grace = tc.get('dropout_grace_s', 0.)
        within_grace = self.last_confirmed_t is not None and 0 < t-self.last_confirmed_t <= grace
        if confirmed:
            if self.prev_confirmed and gap_ok:
                self.confirmed_dwell += dt
            elif within_grace and self.paused_dwell > 0:
                self.confirmed_dwell = self.paused_dwell   # resume; the missing interval is not added
            self.paused_dwell = 0.
            self.last_confirmed_t = t
        else:
            if self.confirmed_dwell > 0:
                self.paused_dwell = self.confirmed_dwell
            if not within_grace:
                self.paused_dwell = 0.
            self.confirmed_dwell = 0.
        self.prev_confirmed = confirmed
        dwell = self.confirmed_dwell

        # object status from the most recent classification (valid for 1 s)
        status, p_unsafe, obj_class, class_conf = 'none',math.nan,'',math.nan
        if self.c.object_gate:
            if self.obj is not None and t - self.obj[0] <= 1.0:
                p_unsafe = self.obj[2] if self.c.calibration else self.obj[1]
                obj_class = self.obj[3]
                conf = self.obj[4]
                class_conf = conf
                if _isnan(conf) and not modern:  # legacy binary feature logs have no category confidence
                    conf = max(p_unsafe, 1 - p_unsafe)
                threshold = self.cfg['calibration']['reject_threshold']
                if (_isnan(conf) or conf < threshold or max(p_unsafe,1-p_unsafe) < threshold
                        or obj_class in ('other','unknown','empty_hand') or self.obj[5] >= .5):
                    status = "uncertain"
                else:
                    status = "unsafe" if p_unsafe >= 0.5 else "safe"
        else:
            status, p_unsafe = "unsafe", 1.0

        hazard = assess_hazard(obj_class,status,self.cfg,self.c.object_gate)
        hazard_source = 'classifier' if hazard.status == 'known_hazard' else 'none'
        detected_hazard = detector_hazard(f, self.cfg) if modern and self.c.object_gate else None
        # A crop trained on eleven categories cannot veto a detector trained on
        # additional classes by returning "other". Keep both predictions visible.
        # A confident contradictory known crop category requests attention only.
        detector_category = strong_detector_category(f, self.cfg) if modern and self.c.object_gate else None
        conflict = bool(detector_category and status in ('safe', 'unsafe') and obj_class != detector_category)
        if detected_hazard is not None:
            hazard = detected_hazard
            hazard_source = 'detector'
        if conflict:
            hazard_source = 'conflict'
        high_category = (not conflict and (
            detected_hazard is not None or (status == 'unsafe' and
            (not modern or hazard.status == 'known_hazard') and p_unsafe >= rc['p_unsafe_danger'])))

        # ---- D3 alert matrix ------------------------------------------------------------
        level, reason = NORMAL, ""
        if near:
            if (high_category and dwell >= rc['dwell_danger_s']
                    and (not modern or f.get('proximity_evidence', False))):
                level, reason = DANGER, ("detected hazard near mouth in image" if hazard_source == 'detector'
                                         else "unsafe object near mouth in image")
            elif status in ("unsafe", "uncertain", "none"):
                level, reason = CAUTION, f"{status} object near mouth"
            elif modern:
                level, reason = CAUTION, "object near mouth; category does not establish safety"
            elif status == "safe" and dwell >= rc["dwell_safe_caution_s"]:
                level, reason = CAUTION, "safe object at mouth for long"   # safe objects cap at caution
        elif (self.c.approach and d is not None and status == "unsafe" and ttc < rc["ttc_caution_s"]
              and d < self.cfg["cascade"]["d_far"]):
            level, reason = CAUTION, "unsafe object approaching"
        candidate_active = (modern and valid and visible and not f.get('association_ambiguous',False)
                and f.get('hand_ok',False) and f.get('detector_ok',False)
                and object_present and not _isnan(candidate_d) and candidate_d <= tc['d_enter'])
        candidate_ready = self.candidate_vote(t,candidate_active)
        renew_existing = bool(modern and self.event_active and self.last_alert_t is not None and candidate_active)
        if (candidate_ready or renew_existing) and level < CAUTION:
            level,reason = CAUTION,'possible held object near mouth; localization uncertain'
        recent_alert = self.last_alert_t is not None and 0 <= t-self.last_alert_t <= sc['lost_hold_s']
        if lost_near and level < CAUTION and (not modern or recent_alert):
            level, reason = CAUTION, "mouth hidden by hand"
        # ---- stable CAUTION -------------------------------------------------------------
        # Briefly retain an alert started by a visible object. A hand remaining
        # over the face never renews that evidence or extends its expiry.
        hold_s = rc.get('caution_hold_s', 0.)
        if modern and (not valid or not visible or not object_present):
            hold_s = min(hold_s,sc['lost_hold_s'])
        current_object_evidence = not modern or (valid and visible and object_present and not _isnan(d_raw))
        if level >= CAUTION and current_object_evidence:
            self.last_alert_t = t
        departed = visible and d is not None and d > tc['d_exit']
        stop = departed or not f.get('source_ok', True) or f.get('person_left', False)
        caution_held = bool(modern and lost_near and level >= CAUTION)
        if stop:
            self.last_alert_t = None
        elif level < CAUTION and self.last_alert_t is not None and hold_s > 0:
            if t - self.last_alert_t <= hold_s:
                level, reason = CAUTION, 'caution held: object was just at the mouth'
                caution_held = True
        # Preserve recent risk, but do not accumulate dwell from held coordinates.
        missing_evidence = not visible or (modern and not f.get('proximity_evidence',False))
        held = (self.c.lost_landmark and missing_evidence and self.last_evidence_t is not None
                and f.get('source_ok', True) and not f.get('person_left',False)
                and 0 <= t - self.last_evidence_t <= sc['lost_hold_s'])
        if held and self.last_level > level:
            level, reason = self.last_level, 'recent evidence held; mouth/object not observable'
        end_reason = ''
        if self.event_active and not (held or caution_held):
            if not f.get('source_ok', True):
                end_reason = 'source_disconnected'
            elif f.get('person_left', False):
                end_reason = 'person_left_frame'
            elif (not visible or f.get('association_ambiguous', False) or (modern and not object_present)
                  or (self.event_reliable and missing_evidence)):
                end_reason = 'observation_lost'
            elif level == NORMAL:
                end_reason = 'observed_departure' if not _isnan(d_raw) and d_raw > tc['d_exit'] else 'risk_changed'
        if end_reason:
            self.event_active = False
            self.event_reliable = False
        elif level > NORMAL:
            self.event_active = True
            self.event_reliable = self.event_reliable or (modern and f.get('proximity_evidence',False))

        # Cooldown suppresses repeat notifications, never the displayed risk.
        notify = False
        if level == DANGER:
            notify = not self.danger_on and (not self.c.cooldown or t >= self.danger_block_until)
            self.danger_on = True
        elif self.danger_on:
            self.danger_on = False
            self.danger_block_until = t + tc["cooldown_s"]

        self.last_level = level
        if modern and level == NORMAL and not object_present and visible and f.get('hand_ok'):
            hand_d = f.get(f'd_{self.norm}')
            if not _isnan(hand_d) and hand_d <= tc['d_exit']:
                reason = 'hand near face; no localized object evidence'
        return {"t": t, "level": level, "reason": reason, "d_raw": d_raw, "d": d if d is not None else math.nan,
                'risk_distance_source':('object' if not _isnan(d_raw) else 'none') if modern else 'hand',
                "speed": speed, "ttc": ttc, "near_frame": near_frame, "near": near, "dwell": dwell,
                "obj_status": status, "p_unsafe": p_unsafe, "obj_class": obj_class, "lost_near": lost_near,
                "notify": notify, 'observation': quality, 'observation_reasons': quality_reasons,
                'obj_conf': class_conf,
                'image_observation':image_quality, 'image_observation_reasons':image_reasons,
                'validation_status':'validated' if f.get('calibrated',False) else 'unvalidated',
                'object_scope':'complete' if f.get('object_coverage',False) else 'limited',
                'hazard_status':hazard.status, 'hazard_type':hazard.kind,
                'hazard_source':hazard_source,
                'hazard_policy_version':hazard.policy_version if modern else 'legacy-dataset-v1',
                'last_evidence_t': self.last_evidence_t, 'held': held or caution_held, 'end_reason': end_reason,
                'green': level == NORMAL and quality == 'observable'}
