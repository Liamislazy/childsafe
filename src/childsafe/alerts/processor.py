"""Per-frame alert processor: smoothing (B2), lost-landmark handling (B2), approach speed (B3-lite),
temporal rule (D1), cascade gating (D2) and the alert matrix (D3).

The same class runs live (scripts/run_webcam.py) and offline on logged features
(scripts/replay.py), so every threshold and every ablation (E4) can be evaluated on
recorded feature logs without touching video again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from childsafe.alerts.temporal import Hysteresis, NofM
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
        n, m = tc["n_of_m"]
        self.vote = NofM(n, m) if self.c.n_of_m else NofM(1, 1)
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
        return not _isnan(d) and d < cc["d_far"] and self.frame % cc["classify_every_n"] == 0

    # ---- main step --------------------------------------------------------------------
    def step(self, f: dict, obj: dict | None = None) -> dict:
        """f: feature dict from GeometryExtractor; obj: ObjectClassifier output for this frame, if run."""
        t = f["t"]
        tc, sc, rc = self.cfg["temporal"], self.cfg["smoothing"], self.cfg["rules"]
        self.frame += 1
        if obj is not None and not _isnan(obj.get("p_unsafe_cal")):
            self.obj = (t, obj["p_unsafe_raw"], obj["p_unsafe_cal"], obj.get("obj_class", ""))

        d_raw = f.get(f"d_{self.norm}")
        lost_near = False
        if _isnan(d_raw):
            d = None
            if self.c.lost_landmark and self.last_face_t is not None and not f.get("face_ok"):
                gap = t - self.last_face_t
                was_near = self.last_d is not None and self.last_d < tc["d_exit"]
                hand_at_old_mouth = not _isnan(f.get("d_last_mouth")) and f["d_last_mouth"] < tc["d_exit"]
                if f.get("hand_ok") and ((was_near and gap <= sc["lost_hold_s"]) or
                                         (hand_at_old_mouth and gap <= sc.get("lost_max_s", 3.0))):
                    lost_near = True
                    if gap <= sc["lost_hold_s"]:
                        d = self.last_d   # hold last state so the temporal rule keeps counting
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
        near = self.vote(near_frame)
        if near and self.near_since is None:
            self.near_since = t
        elif not near:
            self.near_since = None
        dwell = t - self.near_since if near else 0.0

        # object status from the most recent classification (valid for 1 s)
        status, p_unsafe, obj_class = "none", math.nan, ""
        if self.c.object_gate:
            if self.obj is not None and t - self.obj[0] <= 1.0:
                p_unsafe = self.obj[2] if self.c.calibration else self.obj[1]
                obj_class = self.obj[3]
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
            if status == "unsafe" and p_unsafe >= rc["p_unsafe_danger"] and dwell >= rc["dwell_danger_s"]:
                level, reason = DANGER, "unsafe object at mouth"
            elif status in ("unsafe", "uncertain", "none"):
                level, reason = CAUTION, f"{status} object near mouth"
            elif status == "safe" and dwell >= rc["dwell_safe_caution_s"]:
                level, reason = CAUTION, "safe object at mouth for long"   # safe objects cap at caution
        elif (self.c.approach and d is not None and status == "unsafe" and ttc < rc["ttc_caution_s"]
              and d < self.cfg["cascade"]["d_far"]):
            level, reason = CAUTION, "unsafe object approaching"
        if lost_near and level < CAUTION:
            level, reason = CAUTION, "mouth hidden by hand"

        # cooldown: after a danger episode ends, a new one within cooldown_s is shown as caution
        if level == DANGER:
            if not self.danger_on and self.c.cooldown and t < self.danger_block_until:
                level, reason = CAUTION, reason + " (cooldown)"
            else:
                self.danger_on = True
        elif self.danger_on:
            self.danger_on = False
            self.danger_block_until = t + tc["cooldown_s"]

        return {"t": t, "level": level, "reason": reason, "d_raw": d_raw, "d": d if d is not None else math.nan,
                "speed": speed, "ttc": ttc, "near_frame": near_frame, "near": near, "dwell": dwell,
                "obj_status": status, "p_unsafe": p_unsafe, "obj_class": obj_class, "lost_near": lost_near}
