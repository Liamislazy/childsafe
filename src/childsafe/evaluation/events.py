"""E1 event-level metrics.

Ground truth: annotated segments (start, end, label) per video, e.g. exported from ELAN or CVAT
to CSV with columns: video,start,end,label (seconds). Labels listed in `positive` are the
dangerous events (default: unsafe_mouthing). Everything else (eating, drinking, nose_scratch,
chin_rest, safe_mouthing, ...) is non-dangerous and only used to break down false alarms.

Definitions (state them in the write-up, they change the numbers):
- predicted event: a run of frames with level >= `min_level`, gaps < merge_s bridged
- detected GT event: some predicted event overlaps [start - tol, end + tol]
- latency: first alert time inside that window minus GT start (negative = early)
- false alarm: predicted event overlapping no positive GT window
- false alarms per hour: false alarms / hours of video NOT covered by positive events
- per-second agreement: fraction of 1-s bins where (any alert) == (overlaps positive GT)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from childsafe.alerts.temporal import frames_to_events

POSITIVE = ("unsafe_mouthing",)


def _overlap(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


def event_metrics(t: np.ndarray, level: np.ndarray, gt: pd.DataFrame, duration: float | None = None,
                  min_level: int = 2, tol: float = 2.0, merge_s: float = 2.0,
                  positive=POSITIVE) -> dict:
    t = np.asarray(t, float)
    active = np.asarray(level) >= min_level
    duration = duration if duration is not None else (t[-1] - t[0] if len(t) else 0.0)
    pred = frames_to_events(t, active, merge_s)
    pos = gt[gt["label"].isin(positive)][["start", "end"]].to_numpy(float)
    neg = gt[~gt["label"].isin(positive)]

    detected, latencies = 0, []
    for s, e in pos:
        w0, w1 = s - tol, e + tol
        m = active & (t >= w0) & (t <= w1)
        if m.any():
            detected += 1
            latencies.append(float(t[m][0] - s))

    fa, fa_by_label = 0, {}
    for ps, pe in pred:
        if not any(_overlap(ps, pe, s - tol, e + tol) for s, e in pos):
            fa += 1
            hit = [r.label for r in neg.itertuples() if _overlap(ps, pe, r.start, r.end)]
            key = hit[0] if hit else "unlabelled"
            fa_by_label[key] = fa_by_label.get(key, 0) + 1

    pos_time = float(sum(e - s for s, e in pos))
    neg_hours = max(duration - pos_time, 1e-9) / 3600

    n_sec = int(math.ceil(duration)) if duration > 0 else 0
    t0 = t[0] if len(t) else 0.0
    pred_sec = np.zeros(n_sec, bool)
    gt_sec = np.zeros(n_sec, bool)
    if n_sec:
        idx = np.clip(((t - t0)).astype(int), 0, n_sec - 1)
        np.logical_or.at(pred_sec, idx[active], True)
        for s, e in pos:
            gt_sec[max(0, int(s - t0)):min(n_sec, int(math.ceil(e - t0)))] = True

    lat = np.array(latencies)
    return {
        "metric_definition":"legacy-tolerant-inferred-exposure-v1",
        "n_gt_events": int(len(pos)), "n_pred_events": len(pred), "detected": detected,
        "event_recall": detected / len(pos) if len(pos) else float("nan"),
        "false_alarms": fa, "false_alarms_per_hour": fa / neg_hours, "false_alarms_by_label": fa_by_label,
        "latency_median_s": float(np.median(lat)) if len(lat) else float("nan"),
        "latency_p90_s": float(np.percentile(lat, 90)) if len(lat) else float("nan"),
        "event_count_error": len(pred) - len(pos),
        "per_second_agreement": float((pred_sec == gt_sec).mean()) if n_sec else float("nan"),
        "hours": duration / 3600, "non_danger_hours": neg_hours, "latencies": latencies,
    }


def pool(results: list[dict]) -> dict:
    """Pool per-video metrics into one (sums counts, recomputes rates, pools latencies)."""
    g = sum(r["n_gt_events"] for r in results)
    det = sum(r["detected"] for r in results)
    fa = sum(r["false_alarms"] for r in results)
    nh = sum(r["non_danger_hours"] for r in results)
    secs = sum(r["hours"] * 3600 for r in results)
    agree = sum(r["per_second_agreement"] * r["hours"] * 3600 for r in results if not math.isnan(r["per_second_agreement"]))
    lat = np.array([x for r in results for x in r["latencies"]])
    return {"n_videos": len(results), "n_gt_events": g, "detected": det, "event_recall": det / g if g else float("nan"),
            "false_alarms": fa, "false_alarms_per_hour": fa / max(nh, 1e-9),
            "per_second_agreement": agree / secs if secs else float("nan"),
            "event_count_error": sum(r["event_count_error"] for r in results),
            "latency_median_s": float(np.median(lat)) if len(lat) else float("nan"),
            "latency_p90_s": float(np.percentile(lat, 90)) if len(lat) else float("nan"),
            "hours": secs / 3600}


def distance_only_levels(d_raw: np.ndarray, threshold: float) -> np.ndarray:
    """Baseline: per-frame danger whenever the raw normalized distance is below threshold."""
    d = np.asarray(d_raw, float)
    return np.where(np.nan_to_num(d, nan=np.inf) < threshold, 2, 0)
