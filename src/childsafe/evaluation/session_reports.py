"""Shared, provenance-checked input and scoring for reporting and dev-only tuning."""
from dataclasses import dataclass
import copy
import itertools
import json
from pathlib import Path

import pandas as pd

from childsafe.alerts.processor import Components
from childsafe.config import load_config
from childsafe.evaluation.product_report import evaluate_session,pool_reports,METRIC_VERSION
from childsafe.evaluation.sessions import validate_sessions,config_hash,file_hash,code_hash


@dataclass
class SessionRecord:
    path: Path
    manifest: dict
    cfg: dict
    features: pd.DataFrame
    annotations: pd.DataFrame


def load_records(root,split):
    paths=sorted(Path(root).glob('*/manifest.json'))
    manifests=[json.loads(p.read_text(encoding='utf8')) for p in paths]
    validate_sessions(manifests)  # include both partitions when checking subject leakage
    records=[]
    for path,m in zip(paths,manifests):
        if m['split']!=split: continue
        cfg=load_config(path.parent/'config.json')
        if config_hash(cfg)!=m['config_sha256']: raise ValueError('session config hash mismatch')
        if code_hash()!=m['code_sha256']: raise ValueError('runtime source changed; replay with the recorded source revision')
        f=pd.read_csv(path.parent/m['features']); gt=pd.read_csv(path.parent/m['annotations'])
        for field in ('config_sha256','classifier_sha256','detector_sha256','code_sha256'):
            if field not in f or f[field].isna().any() or not f[field].eq(m[field]).all():
                raise ValueError('feature provenance does not match session manifest: '+field)
        records.append(SessionRecord(path,m,cfg,f,gt))
    if not records: raise ValueError('no completed sessions for selected split')
    return records


def replay_config(base,override):
    if override is None: return base
    allowed={'rules','temporal','smoothing','calibration'}
    if any(base.get(k)!=override.get(k) for k in set(base)|set(override) if k not in allowed):
        raise ValueError('replay may change alert thresholds only; detector/crop/model changes require new sessions')
    return override


def score(record,cfg=None,**kwargs):
    m=record.manifest
    return evaluate_session(record.features,record.annotations,replay_config(record.cfg,cfg),
                            m['start_s'],m['end_s'],m['annotation_scope'],**kwargs)


def build_report(records,override=None):
    reports=[]
    for record in records:
        m=record.manifest
        result=score(record,override)
        result.update(recording_id=m['recording_id'],condition=m['condition'],
                      config_sha256=config_hash(replay_config(record.cfg,override)),
                      input_config_sha256=m['config_sha256'],
                      input_features_sha256=file_hash(record.path.parent/m['features']),
                      input_annotations_sha256=file_hash(record.path.parent/m['annotations']),
                      baseline=score(record,override,baseline=True))
        result['caution_or_danger']=score(record,override,min_level=1)
        result['ablations']={name:score(record,override,components=Components.from_off([name]))
                             for name in ('smoothing','n_of_m','object_gate')}
        result['ablation_scope']='alert replay only; detector/ROI changes require matched new sessions'
        reports.append(result)
    conditions={name:pool_reports([r for r in reports if r['condition']==name])
                for name in sorted({r['condition'] for r in reports})}
    return dict(metric_definition=METRIC_VERSION,split=records[0].manifest['split'],
                sessions=reports,by_condition=conditions,overall=pool_reports(reports))


def grid_candidates():
    return itertools.product([.25,.3,.35,.4,.5],[2,3,5],[.5,.6,.7,.8],[0.,.3,.5,1.])


def tune(records,split,min_recall=.9,candidates=None):
    if split!='dev' or any(r.manifest['split']!='dev' for r in records):
        raise ValueError('tune only on dev split')
    base=records[0].cfg
    if any(config_hash(r.cfg)!=config_hash(base) for r in records):
        raise ValueError('tuning requires a common frozen runtime configuration')
    results=[]; gap=base['temporal']['d_exit']-base['temporal']['d_enter']
    for d,n,p,dwell in candidates if candidates is not None else grid_candidates():
        cfg=copy.deepcopy(base)
        cfg['temporal'].update(d_enter=d,d_exit=d+gap,required_evidence_s=n/30)
        cfg['rules'].update(p_unsafe_danger=p,dwell_danger_s=dwell)
        result=pool_reports([score(r,cfg) for r in records])
        if not result['events'] or not result['negative_hours']:
            raise ValueError('dev tuning requires positive events and explicitly labeled negative exposure')
        results.append(dict(d_enter=d,required_evidence_s=n/30,p_unsafe_danger=p,dwell_danger_s=dwell,
                            event_recall=result['event_recall'],false_alarms_per_hour=result['false_alarms_per_hour'],
                            latency_median_s=result['latency_median_s'],metric_definition=METRIC_VERSION))
    if not results: raise ValueError('empty threshold grid')
    eligible=[r for r in results if r['event_recall']>=min_recall]
    if eligible:
        best=min(eligible,key=lambda r:(r['false_alarms_per_hour'],r['latency_median_s'] or 0.))
    else:
        best=min(results,key=lambda r:(-r['event_recall'],r['false_alarms_per_hour']))
    cfg=copy.deepcopy(base)
    cfg['temporal'].update(d_enter=best['d_enter'],d_exit=best['d_enter']+gap,
                           required_evidence_s=best['required_evidence_s'])
    cfg['rules'].update(p_unsafe_danger=best['p_unsafe_danger'],dwell_danger_s=best['dwell_danger_s'])
    return cfg,dict(metric_definition=METRIC_VERSION,split='dev',minimum_recall_met=bool(eligible),
                    selected=best,candidates=results,
                    recording_ids=[r.manifest['recording_id'] for r in records])
