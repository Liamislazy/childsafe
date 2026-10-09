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
        self.w.writerow(["wall_time", "start_s", "end_s", "level", "max_p_unsafe", "obj_class", "reason", 'person_id', 'end_reason'])
        self.cur = None
        self.people = {}

    def update(self, out: dict):
        if 'persons' in out:
            for person in out['persons']:
                self.update(person)
            return
        pid = out.get('person_id', 0)
        self.cur = self.people.get(pid)
        lvl = out["level"]
        if self.cur and (lvl != self.cur['level'] or out.get('end_reason')):
            self._flush(out['t'], out.get('end_reason') or 'risk_changed')
        if lvl != NORMAL and self.cur is None and not out.get('end_reason'):
            self.cur = {"start": out["t"], "level": lvl, "p": -1.0, "cls": "", "reason": out["reason"], 'person_id': pid}
        if self.cur is not None:
            p = out["p_unsafe"]
            if not math.isnan(p) and p > self.cur["p"]:
                self.cur["p"], self.cur["cls"] = p, out["obj_class"]
        self.people[pid] = self.cur

    def _flush(self, t_end: float, end_reason='risk_changed'):
        c = self.cur
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{c['start']:.2f}", f"{t_end:.2f}",
                         LEVEL_NAMES[c["level"]], f"{c['p']:.3f}" if c["p"] >= 0 else "", c["cls"], c["reason"], c['person_id'], end_reason])
        self.f.flush()
        self.cur = None

    def close(self, t_end: float | None = None, reason='source_disconnected'):
        for current in self.people.values():
            if current is not None:
                self.cur = current
                self._flush(t_end if t_end is not None else current['start'], reason)
        self.f.close()
