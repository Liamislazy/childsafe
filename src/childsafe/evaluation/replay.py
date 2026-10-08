"""Re-run the alert processor on a logged feature CSV (from extract_features.py or run_webcam.py --log-features)."""
from __future__ import annotations

import math

import pandas as pd

from childsafe.alerts.processor import AlertProcessor, Components


def replay(features: pd.DataFrame, cfg: dict, components: Components | None = None) -> pd.DataFrame:
    proc = AlertProcessor(cfg, components)
    rows = []
    for f in features.to_dict("records"):
        obj = None
        if proc.wants_classification(f) and not _nan(f.get("p_unsafe_cal")):
            obj = {"p_unsafe_raw": f["p_unsafe_raw"], "p_unsafe_cal": f["p_unsafe_cal"],
                   "obj_class": f.get("obj_class", "")}
        rows.append(proc.step(f, obj))
    return pd.DataFrame(rows)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
