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
from childsafe.alerts.observation import observation
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
        self.reset()

    def reset(self):
        self.filter.reset(); self.speed_filter.reset(); self.hyst.reset(); self.vote.reset()
        self.frame = 0
        self.prev_d = self.prev_t = None
        self.last_face_t = self.last_d = None
        self.near_since = None
        self.obj = None            # (t, p_unsafe_raw, p_unsafe_cal, class)
        self.danger_on = False
        self.danger_block_until = -math.inf
        self.last_evidence_t = None
        self.last_level = NORMAL
        self.event_active = False
        self.last_step_t = None
        self.confirmed_dwell = 0.
        self.prev_confirmed = False
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
        t = f["t"]
        tc, sc, rc = self.cfg["temporal"], self.cfg["smoothing"], self.cfg["rules"]
        self.frame += 1
        dt = t-self.last_step_t if self.last_step_t is not None else 0.
        gap_ok = 0 < dt <= tc.get('max_gap_s',.25)
        if self.last_step_t is not None and not gap_ok:
            self.filter.reset(); self.speed_filter.reset(); self.hyst.reset(); self.vote.reset()
            self.prev_d = self.prev_t = None
            self.confirmed_dwell = 0.
            self.prev_confirmed = False
            self.obj = None
        self.last_step_t = t
        quality, quality_reasons = observation(f)
        modern = 'schema_version' in f
        valid = f.get('frame_valid', True) and f.get('source_ok', True)
        visible = bool(f.get('mouth_visible', f.get('face_ok'))) and valid
        if modern and (not valid or f.get('association_ambiguous')):
            obj = None
        if obj is not None and not _isnan(obj.get("p_unsafe_cal")):
            self.last_classify_t = t
            self.obj = (t, obj["p_unsafe_raw"], obj["p_unsafe_cal"], obj.get("obj_class", ""),
                        obj.get("obj_conf", math.nan))

        d_raw = f.get(f"d_{self.norm}")
        if modern and f.get('proximity_evidence', False):
            d_raw = f.get('object_d_relative')
        if not visible or (modern and f.get('association_ambiguous')):
            d_raw = math.nan
        if visible and not _isnan(d_raw):
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
        near = self.vote(t,near_frame)
        if near and self.near_since is None:
            self.near_since = t
        elif not near:
            self.near_since = None
        confirmed = near and visible and (not modern or f.get('proximity_evidence',False))
        if confirmed and self.prev_confirmed and gap_ok:
            self.confirmed_dwell += dt
        elif not confirmed:
            self.confirmed_dwell = 0.
        self.prev_confirmed = confirmed
        dwell = self.confirmed_dwell

        # object status from the most recent classification (valid for 1 s)
        status, p_unsafe, obj_class = "none", math.nan, ""
        if self.c.object_gate:
            if self.obj is not None and t - self.obj[0] <= 1.0:
                p_unsafe = self.obj[2] if self.c.calibration else self.obj[1]
                obj_class = self.obj[3]
                conf = self.obj[4]
                if _isnan(conf):  # legacy binary feature logs have no category confidence
                    conf = max(p_unsafe, 1 - p_unsafe)
                if conf < self.cfg["calibration"]["reject_threshold"]:
                    status = "uncertain"
                else:
                    status = "unsafe" if p_unsafe >= 0.5 else "safe"
        else:
            status, p_unsafe = "unsafe", 1.0

        # ---- D3 alert matrix ------------------------------------------------------------
        level, reason = NORMAL, ""
        if near:
            if (status == 'unsafe' and p_unsafe >= rc['p_unsafe_danger'] and dwell >= rc['dwell_danger_s']
                    and (not modern or f.get('proximity_evidence', False))):
                level, reason = DANGER, "unsafe object near mouth in image"
            elif status in ("unsafe", "uncertain", "none"):
                level, reason = CAUTION, f"{status} object near mouth"
            elif status == "safe" and dwell >= rc["dwell_safe_caution_s"]:
                level, reason = CAUTION, "safe object at mouth for long"   # safe objects cap at caution
        elif (self.c.approach and d is not None and status == "unsafe" and ttc < rc["ttc_caution_s"]
              and d < self.cfg["cascade"]["d_far"]):
            level, reason = CAUTION, "unsafe object approaching"
        if lost_near and level < CAUTION:
            level, reason = CAUTION, "mouth hidden by hand"
        # Preserve recent risk, but do not accumulate dwell from held coordinates.
        held = (self.c.lost_landmark and not visible and self.last_evidence_t is not None
                and f.get('source_ok', True) and not f.get('person_left',False)
                and 0 <= t - self.last_evidence_t <= sc['lost_hold_s'])
        if held and self.last_level > level:
            level, reason = self.last_level, 'recent evidence held; mouth not observable'
        end_reason = ''
        if self.event_active and not held:
            if not f.get('source_ok', True):
                end_reason = 'source_disconnected'
            elif f.get('person_left', False):
                end_reason = 'person_left_frame'
            elif not visible or f.get('association_ambiguous', False):
                end_reason = 'observation_lost'
            elif level == NORMAL:
                end_reason = 'observed_departure' if not _isnan(d_raw) and d_raw > tc['d_exit'] else 'risk_changed'
        if end_reason:
            self.event_active = False
        elif level > NORMAL:
            self.event_active = True

        # Cooldown suppresses repeat notifications, never the displayed risk.
        notify = False
        if level == DANGER:
            notify = not self.danger_on and (not self.c.cooldown or t >= self.danger_block_until)
            self.danger_on = True
        elif self.danger_on:
            self.danger_on = False
            self.danger_block_until = t + tc["cooldown_s"]

        self.last_level = level
        return {"t": t, "level": level, "reason": reason, "d_raw": d_raw, "d": d if d is not None else math.nan,
                "speed": speed, "ttc": ttc, "near_frame": near_frame, "near": near, "dwell": dwell,
                "obj_status": status, "p_unsafe": p_unsafe, "obj_class": obj_class, "lost_near": lost_near,
                "notify": notify, 'observation': quality, 'observation_reasons': quality_reasons,
                'last_evidence_t': self.last_evidence_t, 'held': held, 'end_reason': end_reason,
                'green': level == NORMAL and quality == 'observable'}
