"""B2: scalar smoothing filters with timestamps.

One Euro filter: Casiez, Roussel & Vogel, CHI 2012. Low jitter when still, low lag when moving.
"""
from __future__ import annotations

import math


class NoFilter:
    def __call__(self, x: float, t: float) -> float:
        return x

    def reset(self):
        pass


class EMA:
    def __init__(self, alpha: float = 0.4):
        self.alpha, self.y = alpha, None

    def __call__(self, x: float, t: float) -> float:
        self.y = x if self.y is None else self.alpha * x + (1 - self.alpha) * self.y
        return self.y

    def reset(self):
        self.y = None


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def make_filter(cfg: dict):
    m = cfg["method"]
    if m == "none":
        return NoFilter()
    if m == "ema":
        return EMA(cfg["ema_alpha"])
    if m == "one_euro":
        return OneEuro(cfg["one_euro_min_cutoff"], cfg["one_euro_beta"])
    raise ValueError(m)
