"""D1 building blocks: hysteresis, N-of-M voting, event merging."""
from __future__ import annotations

from collections import deque

import numpy as np


class Hysteresis:
    """Enter 'near' when d < enter, leave only when d > exit (enter < exit)."""

    def __init__(self, enter: float, exit: float):
        assert enter <= exit, "hysteresis needs enter <= exit"
        self.enter, self.exit, self.state = enter, exit, False

    def __call__(self, d: float) -> bool:
        if d is None or np.isnan(d):
            self.state = False
        elif self.state:
            self.state = d <= self.exit
        else:
            self.state = d < self.enter
        return self.state

    def reset(self):
        self.state = False


class NofM:
    """True when at least n of the last m inputs were True."""

    def __init__(self, n: int, m: int):
        assert 1 <= n <= m
        self.n, self.buf = n, deque(maxlen=m)

    def __call__(self, x: bool) -> bool:
        self.buf.append(bool(x))
        return sum(self.buf) >= self.n

    def reset(self):
        self.buf.clear()


def frames_to_events(t: np.ndarray, active: np.ndarray, merge_s: float = 0.0) -> list[tuple[float, float]]:
    """Turn a per-frame boolean series into (start, end) events; gaps shorter than merge_s are bridged."""
    t = np.asarray(t, float)
    active = np.asarray(active, bool)
    events = []
    start = None
    for i in range(len(t)):
        if active[i] and start is None:
            start = t[i]
        if start is not None and (not active[i] or i == len(t) - 1):
            end = t[i] if active[i] else t[i - 1]
            events.append([start, end])
            start = None
    merged: list[list[float]] = []
    for s, e in events:
        if merged and s - merged[-1][1] <= merge_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]
