"""Evaluate complete recordings before any conditional observation analysis."""
import math
import numpy as np
from childsafe.evaluation.events import event_metrics, pool
from childsafe.evaluation.replay import replay, aggregate_results
from childsafe.evaluation.coverage import observation_metrics


def recording_report(features, gt, cfg, components=None, min_level=2, tol=2.):
    out = replay(features,cfg,components)
    aggregate = aggregate_results(out)
    results = []
    observed_hits = observed_events = 0
    sessions = aggregate.session_id.unique()
    if len(sessions) > 1 and 'session_id' not in gt:
        raise ValueError('annotations need session_id for recordings with reconnects')
    for session in sessions:
        frames = aggregate[aggregate.session_id == session]
        ann = gt[gt.session_id == session] if 'session_id' in gt else gt
        t = frames.t.to_numpy(float)
        duration = (t[-1]-t[0]) if len(t) else 0.
        if 'person_id' in ann:
            ids = set(ann.person_id) | set(out[out.session_id == session].person_id)
            groups = [(out[(out.session_id == session) & (out.person_id == pid)],ann[ann.person_id == pid])
                      for pid in ids if pid != 0]
        else:
            groups = [(frames,ann)]  # explicitly camera-level legacy annotations
        for group,annotations in groups:
            # An annotated person never detected still contributes missed events to total recall.
            levels = group.level.to_numpy() if len(group) else np.zeros(len(t),int)
            times = group.t.to_numpy(float) if len(group) else t
            results.append(event_metrics(times,levels,annotations,duration,min_level,tol,
                                         cfg['temporal']['merge_s']))
            for event in annotations[annotations.label == 'unsafe_mouthing'].itertuples():
                segment = group[(group.t >= event.start) & (group.t <= event.end)]
                fully_observed = (len(segment) >= 2 and (segment.observation == 'observable').all() and
                                  np.diff(segment.t).max(initial=0) <= cfg['temporal']['max_gap_s'] and
                                  segment.t.iloc[0]-event.start <= cfg['temporal']['max_gap_s'] and
                                  event.end-segment.t.iloc[-1] <= cfg['temporal']['max_gap_s'])
                if fully_observed:
                    observed_events += 1
                    window = group[(group.t >= event.start-tol) & (group.t <= event.end+tol)]
                    observed_hits += int(((window.level >= min_level) & (window.observation == 'observable')).any())
    metrics = pool(results)
    metrics['latencies'] = [x for r in results for x in r['latencies']]
    metrics['non_danger_hours'] = sum(r['non_danger_hours'] for r in results)
    metrics['n_pred_events'] = sum(r['n_pred_events'] for r in results)
    metrics.update(observation_metrics(out,cfg))
    metrics.update(observed_event_recall=observed_hits/observed_events if observed_events else math.nan,
                   n_fully_observed_events=observed_events,
                   annotation_scope='person' if 'person_id' in gt else 'camera',
                   latency_scope='alert relative to annotated event; frame age is internal only')
    return metrics
