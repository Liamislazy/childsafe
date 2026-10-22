"""Bind browser numerical exports to a consented, frozen scenario manifest."""
import argparse
import json
from pathlib import Path
import pandas as pd
from childsafe.config import load_config
from childsafe.evaluation.product_report import evaluate_session,METRIC_VERSION
from childsafe.evaluation.sessions import config_hash,code_hash,file_hash,validate_sessions


def finalize(folder,features,annotations,metadata):
    folder=Path(folder); manifest_path=folder/'manifest.json'
    m=json.loads(manifest_path.read_text(encoding='utf8'))
    validate_sessions([json.loads(p.read_text(encoding='utf8')) for p in folder.parent.glob('*/manifest.json')])
    if m.get('status')=='complete': raise ValueError('session already finalized; use a new recording ID')
    meta=json.loads(Path(metadata).read_text(encoding='utf8'))
    if meta.get('unfinished_annotation'): raise ValueError('finish the independent annotation before ending the camera session')
    cfg=load_config(folder/'config.json')
    if config_hash(cfg)!=m['config_sha256'] or code_hash()!=m['code_sha256']:
        raise ValueError('frozen configuration or source revision changed')
    fields=('config_sha256','classifier_sha256','detector_sha256','code_sha256')
    if any(meta.get('provenance',{}).get(k)!=m[k] for k in fields):
        raise ValueError('browser models/config/source do not match the prepared session')
    try: f=pd.read_csv(features)
    except pd.errors.EmptyDataError:
        f=pd.DataFrame(columns=['t','session_id','person_id','schema_version',*fields])
    gt=pd.read_csv(annotations)
    if not f.empty:
        if not f.session_id.eq(meta['runtime_session_id']).all(): raise ValueError('mixed browser epochs')
        if any(k not in f or f[k].isna().any() or not f[k].eq(m[k]).all() for k in fields):
            raise ValueError('feature provenance mismatch')
    if 'session_id' in gt and not gt.session_id.eq(meta['runtime_session_id']).all():
        raise ValueError('annotation epoch mismatch')
    evaluate_session(f,gt,cfg,meta['start_s'],meta['end_s'],m['annotation_scope'])
    # Validate everything before replacing the prepared empty templates.
    f.to_csv(folder/m['features'],index=False); gt.to_csv(folder/m['annotations'],index=False)
    labels=f.loc[f.person_id.gt(0),['session_id','person_id','t']].copy()
    labels['mouth_observable']=''; labels['reason']=''
    if m['split']=='dev': labels.to_csv(folder/'observation_labels.csv',index=False)
    m.update(start_s=meta['start_s'],end_s=meta['end_s'],runtime_session_id=meta['runtime_session_id'],
             status='complete',metric_definition=METRIC_VERSION,
             features_sha256=file_hash(folder/m['features']),annotations_sha256=file_hash(folder/m['annotations']),
             annotation_clock=meta.get('annotation_clock','not supplied'),
             # Exports from before video mode carry no field and were camera-only.
             input_source=meta.get('input_source','camera'))
    manifest_path.write_text(json.dumps(m,indent=2),encoding='utf8')
    return folder


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path); p.add_argument('--features',required=True,type=Path)
    p.add_argument('--annotations',required=True,type=Path); p.add_argument('--metadata',required=True,type=Path)
    a=p.parse_args(argv)
    print(finalize(a.folder,a.features,a.annotations,a.metadata))


if __name__=='__main__': main()
