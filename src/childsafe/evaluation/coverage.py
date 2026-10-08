"""Time-weighted observation coverage; legacy logs have unknown coverage."""
import math
import numpy as np


def observation_metrics(out, cfg):
    if out.empty:
        return dict(observation_coverage=math.nan, observation_note='no observations', fps=math.nan)
    if any(any('legacy:' in reason for reason in reasons) for reasons in out.get('observation_reasons', [])):
        return dict(observation_coverage=math.nan,
                    observation_note='không có dữ liệu chất lượng quan sát',fps=math.nan)
    from childsafe.evaluation.replay import aggregate_results
    frames = aggregate_results(out)
    observed = total = 0.
    counts = 0
    for _,group in frames.groupby(['video','session_id'],sort=False):
        t = group.t.to_numpy(float)
        quality = group.observation.to_numpy()
        if len(t) < 2:
            continue
        dt = np.diff(t)
        dt = np.maximum(dt,0)
        total += float(dt.sum())
        usable = (dt <= cfg['temporal'].get('max_gap_s',.25)) & (quality[:-1] == 'observable') & (quality[1:] == 'observable')
        observed += float(dt[usable].sum())
        counts += len(t)-1
    inf = frames.inference_s.dropna().to_numpy(float)
    age = frames.frame_age_s.dropna().to_numpy(float)
    return dict(observation_coverage=observed/total if total else math.nan,
                observation_note='true-to-true observable intervals; long gaps count as unobserved',
                fps=counts/total if total else math.nan,
                inference_median_s=float(np.median(inf)) if len(inf) else math.nan,
                inference_p90_s=float(np.percentile(inf,90)) if len(inf) else math.nan,
                internal_age_median_s=float(np.median(age)) if len(age) else math.nan,
                internal_age_p90_s=float(np.percentile(age,90)) if len(age) else math.nan,
                duration_s=total, observable_s=observed)
