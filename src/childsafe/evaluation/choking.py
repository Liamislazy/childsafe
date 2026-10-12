"""Metadata-only evaluation of a FUTURE choking model, never a choking detector.

Mouthing risk levels are deliberately not accepted as predictions. Simulations and
clinical references are reported separately. No videos or audio are opened here.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

PREDICTION_COLUMNS = {'recording_id', 'subject_id', 't', 'choking_alarm', 'choking_observation'}
ANNOTATION_COLUMNS = {'recording_id', 'subject_id', 'start', 'end', 'diagnosis', 'cause', 'review_reference'}
RECORDING_COLUMNS = {'recording_id', 'duration_s', 'evidence_kind', 'age_group', 'viewpoint'}


def _require(df, columns, name):
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f'{name} missing {sorted(missing)}; unsafe_mouthing/DANGER is not a choking prediction')
    if df[list(columns)].isna().any().any():
        raise ValueError(f'{name} has missing required metadata')


def _summarize(rows):
    positives = sum(r['n_choking_events'] for r in rows)
    detected = sum(r['detected_choking_events'] for r in rows)
    negatives = sum(r['known_non_choking_s'] for r in rows)
    false_alarms = sum(r['false_alarm_episodes'] for r in rows)
    latencies = [x for r in rows for x in r['latencies_s']]
    duration = sum(r['duration_s'] for r in rows)
    observable = sum(r['observable_s'] for r in rows)
    toys = sum(r['n_toy_choking_events'] for r in rows)
    return dict(n_subject_sessions=len(rows), n_choking_events=positives,
                detected_choking_events=detected, event_recall=detected/positives if positives else None,
                n_toy_choking_events=toys,
                toy_event_recall=sum(r['detected_toy_choking_events'] for r in rows)/toys if toys else None,
                false_alarm_episodes=false_alarms, known_non_choking_s=negatives,
                false_alarms_per_known_negative_hour=false_alarms/(negatives/3600) if negatives else None,
                duration_s=duration, observable_s=observable,
                observation_coverage=observable/duration if duration else None,
                latency_median_s=float(np.median(latencies)) if latencies else None,
                latency_p90_s=float(np.percentile(latencies, 90)) if latencies else None,
                unknown_reference_alarm_episodes=sum(r['unknown_reference_alarm_episodes'] for r in rows))


def choking_report(predictions, annotations, recordings, max_gap_s=.25):
    """Evaluate anonymous subject-session metadata, including completely missed subjects.

    Inputs describe fused subject sessions, not unrelated camera-local tracker IDs.
    Clinical/simulation provenance is a declared input; this function cannot verify
    a clinician, model validity, or the authenticity of a medical diagnosis.
    Reference intervals are half-open [start,end); unknown/unlabelled time is NOT negative.
    """
    for df, columns, name in ((predictions, PREDICTION_COLUMNS, 'predictions'),
                              (annotations, ANNOTATION_COLUMNS, 'annotations'),
                              (recordings, RECORDING_COLUMNS, 'recordings')):
        _require(df, columns, name)
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('max_gap_s must be positive and finite')
    if recordings.empty or recordings.recording_id.duplicated().any():
        raise ValueError('recordings require unique IDs and nonempty metadata')
    if not set(recordings.evidence_kind).issubset({'simulation', 'clinical'}):
        raise ValueError('evidence_kind must be simulation or clinical')
    if not set(annotations.diagnosis).issubset({'choking', 'non_choking', 'uncertain'}):
        raise ValueError('diagnosis must be choking, non_choking or uncertain')
    if not set(annotations.cause).issubset({'toy', 'food', 'other', 'unknown'}):
        raise ValueError('cause must be toy, food, other or unknown')
    if not set(predictions.choking_observation).issubset({'observable', 'degraded', 'unobservable'}):
        raise ValueError('invalid choking_observation; mouthing visibility is insufficient')
    if not predictions.choking_alarm.isin([0, 1, False, True]).all():
        raise ValueError('choking_alarm must be explicitly boolean; no missing predictions mapped to healthy')
    if predictions.duplicated(['recording_id', 'subject_id', 't']).any():
        raise ValueError('duplicate subject/time: synchronize and fuse cameras before evaluation')
    meta = recordings.set_index('recording_id')
    ids = set(meta.index)
    if not set(predictions.recording_id).issubset(ids) or not set(annotations.recording_id).issubset(ids):
        raise ValueError('unknown recording ID')
    for recording_id, row in meta.iterrows():
        if not math.isfinite(row.duration_s) or row.duration_s <= 0:
            raise ValueError('duration_s must be positive and finite')
        if not str(row.age_group).strip() or not str(row.viewpoint).strip():
            raise ValueError('age_group/viewpoint must be declared, including unknown if not verified')
        refs = annotations[annotations.recording_id == recording_id]
        if row.evidence_kind == 'clinical' and not refs.review_reference.astype(str).str.strip().ne('').all():
            raise ValueError('clinical reference requires an independent review_reference')
    pairs = set(zip(predictions.recording_id, predictions.subject_id)) | set(zip(annotations.recording_id, annotations.subject_id))
    rows = []
    for recording_id, subject_id in sorted(pairs):
        session = meta.loc[recording_id]
        duration = float(session.duration_s)
        frames = predictions[(predictions.recording_id == recording_id) & (predictions.subject_id == subject_id)].sort_values('t')
        refs = annotations[(annotations.recording_id == recording_id) & (annotations.subject_id == subject_id)].sort_values('start')
        t = frames.t.to_numpy(float)
        if not np.isfinite(t).all() or (t < 0).any() or (t >= duration).any():
            raise ValueError('prediction timestamps must lie inside the subject session')
        starts, ends = refs.start.to_numpy(float), refs.end.to_numpy(float)
        if (not np.isfinite(starts).all() or not np.isfinite(ends).all() or
                (starts < 0).any() or (ends > duration).any() or (ends <= starts).any() or
                (starts[1:] < ends[:-1]).any()):
            raise ValueError('reference intervals must be valid, nonoverlapping and inside session')
        alarm = frames.choking_alarm.to_numpy(bool)
        negative_at_frame = np.zeros(len(t), bool)
        known_at_frame = np.zeros(len(t), bool)
        positive, detected, toys, detected_toys, negative_time, latencies = 0, 0, 0, 0, 0., []
        for ref in refs.itertuples():
            inside = (t >= ref.start) & (t < ref.end)
            if ref.diagnosis == 'choking':
                positive += 1
                toy = ref.cause == 'toy'; toys += int(toy)
                hits = t[inside & alarm]
                if len(hits):
                    detected += 1; detected_toys += int(toy); latencies.append(float(hits[0]-ref.start))
                known_at_frame |= inside
            elif ref.diagnosis == 'non_choking':
                negative_time += float(ref.end-ref.start)
                negative_at_frame |= inside; known_at_frame |= inside
        # An alarm episode intersecting known negative observations counts as false,
        # even if the same long-running alarm also overlaps a positive event.
        episodes = []
        for index in np.flatnonzero(alarm):
            if not episodes or index != episodes[-1][-1]+1 or t[index]-t[episodes[-1][-1]] > max_gap_s:
                episodes.append([int(index)])
            else:
                episodes[-1].append(int(index))
        false_alarms = sum(bool(negative_at_frame[indexes].any()) for indexes in episodes)
        unknown_alarms = sum(bool((~known_at_frame[indexes]).all()) for indexes in episodes)
        dt = np.diff(t)
        observation = frames.choking_observation.to_numpy()
        usable = ((dt > 0) & (dt <= max_gap_s) & (observation[:-1] == 'observable') &
                  (observation[1:] == 'observable'))
        rows.append(dict(recording_id=recording_id, subject_id=subject_id,
                         evidence_kind=session.evidence_kind, age_group=session.age_group, viewpoint=session.viewpoint,
                         duration_s=duration, observable_s=float(dt[usable].sum()),
                         n_choking_events=positive, detected_choking_events=detected,
                         n_toy_choking_events=toys, detected_toy_choking_events=detected_toys,
                         known_non_choking_s=negative_time, false_alarm_episodes=false_alarms,
                         unknown_reference_alarm_episodes=unknown_alarms, latencies_s=latencies))
    # Never pool clinical and simulated outcomes into a purported clinical score.
    by_kind = {}
    for kind in ('simulation', 'clinical'):
        group = [r for r in rows if r['evidence_kind'] == kind]
        by_kind[kind] = _summarize(group) if group else None
    strata = {}
    for kind, age, view in sorted({(r['evidence_kind'], r['age_group'], r['viewpoint']) for r in rows}):
        strata[f'{kind}/{age}/{view}'] = _summarize([r for r in rows if
                                          (r['evidence_kind'], r['age_group'], r['viewpoint']) == (kind, age, view)])
    return dict(by_evidence_kind=by_kind, by_age_and_viewpoint=strata, subject_sessions=rows,
                clinical_validity='not established by this metadata evaluator',
                definitions={'latency': 'first choking alarm inside reference event, no pre-event tolerance',
                             'negative_time': 'only explicitly reviewed non_choking intervals',
                             'false_alarm_episode': 'an alarm run with any sampled known-negative time',
                             'coverage': 'consecutive observable intervals <= max_gap_s over full subject-session duration'},
                limitations=['Simulation cannot establish choking detection in children.',
                             'Reference provenance and diagnosis require independent audit.',
                             'An absent age/view stratum is not evaluated; it is not evidence of universal coverage.'])
