import importlib.util
import json
from pathlib import Path
import pandas as pd
import pytest
from childsafe.config import load_config
from childsafe.evaluation.sessions import create_session
from childsafe.evaluation.session_reports import load_records,build_report


def script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/name
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path):
    model=tmp_path/'model.pt'; model.write_bytes(b'fixture')
    cfg=load_config(); cfg['detector']['weights']=str(model)
    root=tmp_path/'sessions'
    folder=create_session(root,'dev01','dev',['adult1'],'elevated',cfg,model,consent_confirmed=True)
    m=json.loads((folder/'manifest.json').read_text())
    provenance={k:m[k] for k in ('config_sha256','classifier_sha256','detector_sha256','code_sha256')}
    features=tmp_path/'camera.csv'; annotations=tmp_path/'annotations.csv'; metadata=tmp_path/'session.json'
    pd.DataFrame([dict(t=t,session_id=1,person_id=1,schema_version=2,mouth_visible=False,
                       face_ok=False,hand_ok=False,d_face_width=1.,**provenance) for t in (1.1,1.2)]).to_csv(features,index=False)
    pd.DataFrame([dict(start=1,end=2,label='non_event',session_id=1)]).to_csv(annotations,index=False)
    metadata.write_text(json.dumps(dict(provenance=provenance,start_s=1,end_s=2,runtime_session_id=1,unfinished_annotation=False)))
    return root,folder,features,annotations,metadata


def test_targeted_plan_is_reproducible_and_contains_no_invented_results():
    plan=script('plan_camera_trials.py').plan
    rows=plan(); assert rows==plan() and len(rows)==40
    assert {r['split'] for r in rows}=={'dev','test'}
    assert {r['people'] for r in rows}=={1,2,4}
    assert {r['resolution'] for r in rows}=={'480p','720p','1080p'}
    assert all(r['negative_seconds_observed']==r['positive_events_observed']=='' for r in rows)


def test_finalize_exports_can_be_reported_and_detect_later_tampering(tmp_path):
    root,folder,f,gt,meta=fixture(tmp_path)
    finalize=script('finalize_session.py').finalize
    finalize(folder,f,gt,meta)
    report=build_report(load_records(root,'dev'))
    assert report['overall']['negative_hours']==1/3600
    assert 'mouth_observable' in pd.read_csv(folder/'observation_labels.csv')
    with pytest.raises(ValueError,match='already finalized'): finalize(folder,f,gt,meta)
    with (folder/'annotations.csv').open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError,match='content changed'): load_records(root,'dev')


@pytest.mark.parametrize('bad',['open_marker','late_frame','wrong_model'])
def test_finalize_rejects_bad_exports_before_changing_templates(tmp_path,bad):
    _,folder,f,gt,meta=fixture(tmp_path)
    data=json.loads(meta.read_text())
    if bad=='open_marker': data['unfinished_annotation']=True
    elif bad=='late_frame': data['end_s']=1.15
    else: data['provenance']['classifier_sha256']='wrong'
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError): script('finalize_session.py').finalize(folder,f,gt,meta)
    assert not (folder/'features.csv').exists()
    assert json.loads((folder/'manifest.json').read_text())['start_s'] is None
