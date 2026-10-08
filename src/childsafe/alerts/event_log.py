"""Privacy-preserving event log: only time, level, confidence, object class. No images, no landmarks."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

from childsafe.alerts.processor import LEVEL_NAMES, NORMAL


class EventLogger:
    def __init__(self, log_dir: str | Path):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(log_dir) / f"events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(["wall_time", "start_s", "end_s", "level", "max_p_unsafe", "obj_class", "reason"])
        self.cur = None

    def update(self, out: dict):
        lvl = out["level"]
        if self.cur and lvl != self.cur["level"]:
            self._flush(out["t"])
        if lvl != NORMAL and self.cur is None:
            self.cur = {"start": out["t"], "level": lvl, "p": -1.0, "cls": "", "reason": out["reason"]}
        if self.cur is not None:
            p = out["p_unsafe"]
            if not math.isnan(p) and p > self.cur["p"]:
                self.cur["p"], self.cur["cls"] = p, out["obj_class"]

    def _flush(self, t_end: float):
        c = self.cur
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{c['start']:.2f}", f"{t_end:.2f}",
                         LEVEL_NAMES[c["level"]], f"{c['p']:.3f}" if c["p"] >= 0 else "", c["cls"], c["reason"]])
        self.f.flush()
        self.cur = None

    def close(self, t_end: float | None = None):
        if self.cur is not None:
            self._flush(t_end if t_end is not None else self.cur["start"])
        self.f.close()
