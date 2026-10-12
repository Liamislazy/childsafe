"""Report adult/doll numerical sessions by condition; never loads images/video."""
import argparse
import json
from pathlib import Path
import pandas as pd
from childsafe.config import load_config
from childsafe.evaluation.sessions import validate_sessions,config_hash,file_hash
from childsafe.evaluation.product_report import evaluate_session,summarize
from childsafe.alerts.processor import Components

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path); p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--out',type=Path,default=Path('outputs/product_report.json'))
    a=p.parse_args(); paths=sorted(a.root.glob('*/manifest.json'))
    sessions=[json.loads(f.read_text(encoding='utf8')) for f in paths]
    validate_sessions(sessions); reports=[]
    for path,m in zip(paths,sessions):
        if m['split']!=a.split: continue
        cfg=load_config(path.parent/'config.json')
        if config_hash(cfg)!=m['config_sha256']: raise ValueError('session config hash mismatch')
        f=pd.read_csv(path.parent/m['features']); gt=pd.read_csv(path.parent/m['annotations'])
        result=evaluate_session(f,gt,cfg,m['start_s'],m['end_s'],m['annotation_scope'])
        result.update(recording_id=m['recording_id'],condition=m['condition'],
                      config_sha256=m['config_sha256'],input_features_sha256=file_hash(path.parent/m['features']),
                      baseline=evaluate_session(f,gt,cfg,m['start_s'],m['end_s'],m['annotation_scope'],True))
        result['caution_or_danger']=evaluate_session(f,gt,cfg,m['start_s'],m['end_s'],m['annotation_scope'],min_level=1)
        result['ablations']={name:evaluate_session(f,gt,cfg,m['start_s'],m['end_s'],m['annotation_scope'],
                                                  components=Components.from_off([name]))
                              for name in ('smoothing','n_of_m','object_gate')}
        result['ablation_scope']='alert replay only; detector/ROI changes require matched new sessions'
        reports.append(result)
    if not reports: raise ValueError('no completed sessions for selected split')
    condition={}
    fields=['events','hits','false_alarms','unknown_alarm_episodes','negative_hours','observable_events','observable_hits','observable_s','duration_s']
    for name in sorted({r['condition'] for r in reports}):
        parts=[r for r in reports if r['condition']==name]
        c={k:sum(r[k] for r in parts) for k in fields}; c['latencies']=[v for r in parts for v in r['latencies']]
        c['legacy_quality']=any(r['legacy_quality'] for r in parts)
        condition[name]=summarize(c)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(split=a.split,sessions=reports,by_condition=condition),indent=2,allow_nan=False),encoding='utf8')
    print(a.out)
