"""Profile exported numerical sessions; person rows never count as extra frames."""
import math
import numpy as np


def runtime_profile(features, max_gap_s=.25):
    if features.empty:
        return {'samples':0,'scope':'no samples'}
    keys = [c for c in ('video','session_id','t') if c in features]
    frames = features.groupby(keys,dropna=False).max(numeric_only=True).reset_index()
    intervals = []
    grouping = [c for c in ('video','session_id') if c in frames]
    groups = frames.groupby(grouping,dropna=False) if grouping else [(None,frames)]
    for _,group in groups:
        intervals.extend(np.diff(np.sort(group.t.unique())).tolist())
    gaps = np.asarray(intervals,float)
    result = {'samples':len(frames),'processed_gap_p90_s':float(np.percentile(gaps,90)) if len(gaps) else None,
              'continuity_breaks':int((gaps > max_gap_s).sum()),
              'processed_fps':len(gaps)/gaps.sum() if len(gaps) and gaps.sum() > 0 else None,
              'scope':'processed sample times; not sensor capture rate or clinical latency'}
    result['peak_process_memory_bytes'] = int(frames.peak_process_memory_bytes.max()) if 'peak_process_memory_bytes' in frames and frames.peak_process_memory_bytes.notna().any() else None
    result['capture_slot_overwrites_total'] = int(frames.capture_slot_overwrites.max()) if 'capture_slot_overwrites' in frames and frames.capture_slot_overwrites.notna().any() else None
    result['inference_budget_exceeded_samples']=int((frames.inference_s>max_gap_s).sum()) if 'inference_s' in frames else None
    result['invalid_samples']=int((~frames.frame_valid.astype(bool)).sum()) if 'frame_valid' in frames else None
    for field in ('detector_calls','roi_detector_calls'):
        result[field+'_total']=int(frames[field].sum()) if field in frames else None
    if 'tracked' in frames:
        per_person=features[features.tracked.eq(True)]
        counts=per_person.groupby(keys,dropna=False).person_id.nunique().reindex(frames.set_index(keys).index,fill_value=0)
        result['samples_by_tracked_people']={str(int(k)):int(v) for k,v in counts.value_counts().items()}
    for field in ('detector_s','landmarks_s','classifier_s','quality_s','other_s','inference_s','frame_age_s','decode_s','processed_gap_s'):
        values = frames[field].dropna().to_numpy(float) if field in frames else []
        values = [v for v in values if math.isfinite(v)]
        result[field] = {'median':float(np.median(values)), 'p90':float(np.percentile(values,90))} if len(values) else None
    return result
