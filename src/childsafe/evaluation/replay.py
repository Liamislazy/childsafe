"""Re-run the alert processor on a logged feature CSV (from extract_features.py or run_webcam.py --log-features)."""
from __future__ import annotations

import math
import ast

import pandas as pd

from childsafe.alerts.processor import AlertProcessor, Components


def replay(features: pd.DataFrame, cfg: dict, components: Components | None = None) -> pd.DataFrame:
    processors = {}
    sessions = {}
    rows = []
    for f in features.to_dict("records"):
        video = f.get('video','')
        session = f.get('session_id',0)
        if sessions.get(video) != session:
            processors = {key: value for key,value in processors.items() if key[0] != video}
            sessions[video] = session
        pid = f.get('person_id',0)
        key = (video,pid)
        proc = processors.setdefault(key, AlertProcessor(cfg, components))
        if isinstance(f.get('quality_reasons'),str):
            try:
                reasons = ast.literal_eval(f['quality_reasons'])
                f['quality_reasons'] = reasons if isinstance(reasons,list) else []
            except (SyntaxError,ValueError):
                f['quality_reasons'] = ['invalid quality reasons']
        for name in ('mouth_visible','source_ok','frame_valid','association_ambiguous','detector_ok',
                     'classifier_ok','calibrated','object_coverage','proximity_evidence','object_present','person_left','hand_ok','face_ok','history_reset'):
            if name in f:
                value = f[name]
                f[name] = value is True or value == 1 or str(value).lower() == 'true'
        obj = None
        modern = not _nan(f.get('schema_version'))
        if not modern:
            f.pop('schema_version',None)
        if (modern or proc.wants_classification(f)) and not _nan(f.get("p_unsafe_cal")):
            obj = {"p_unsafe_raw": f["p_unsafe_raw"], "p_unsafe_cal": f["p_unsafe_cal"],
                   'obj_class': f.get('obj_class',''), 'obj_conf': f.get('obj_conf',math.nan),
                   'p_unknown_cal': f.get('p_unknown_cal',0.)}
        out = proc.step(f, obj)
        out.update(person_id=pid, session_id=session, video=video,
                   inference_s=f.get('inference_s',math.nan), frame_age_s=f.get('frame_age_s',math.nan))
        rows.append(out)
    return pd.DataFrame(rows)


def aggregate_results(out: pd.DataFrame) -> pd.DataFrame:
    """One result per source sample; never count person rows as extra elapsed frames."""
    rows = []
    for (video,session,t),group in out.groupby(['video','session_id','t'],sort=False):
        qualities = group['observation']
        quality = ('unobservable' if (qualities == 'unobservable').any() else
                   'degraded' if (qualities == 'degraded').any() else 'observable')
        rows.append(dict(video=video,session_id=session,t=t,level=int(group.level.max()),
                         observation=quality,inference_s=group.inference_s.max(),frame_age_s=group.frame_age_s.max()))
    return pd.DataFrame(rows)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
