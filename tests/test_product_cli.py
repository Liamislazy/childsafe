import json
import runpy
import sys
from pathlib import Path
import pandas as pd
from childsafe.config import load_config
from childsafe.evaluation.sessions import create_session


def test_report_cli_reads_frozen_config_and_writes_finite_json(tmp_path,monkeypatch):
    model=tmp_path/'model.pt'; model.write_bytes(b'fixture')
    cfg=load_config(); cfg['detector']['weights']=str(model)
    folder=create_session(tmp_path,'test01','test',['adult2'],'elevated',cfg,model,consent_confirmed=True)
    m=json.loads((folder/'manifest.json').read_text()); m.update(start_s=0,end_s=1)
    (folder/'manifest.json').write_text(json.dumps(m))
    pd.DataFrame([dict(t=t,session_id=1,person_id=0,schema_version=2,mouth_visible=False,
                      face_ok=False,hand_ok=False,d_face_width=1.,
                      **{key:m[key] for key in ('config_sha256','classifier_sha256','detector_sha256','code_sha256')})
                  for t in (0,.1,.2)]).to_csv(folder/'features.csv',index=False)
    pd.DataFrame([dict(start=0,end=1,label='non_event')]).to_csv(folder/'annotations.csv',index=False)
    output=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['report_product_sessions',str(tmp_path),'--split','test','--out',str(output)])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/report_product_sessions.py'),run_name='__main__')
    report=json.loads(output.read_text())
    assert report['by_condition']['elevated']['negative_hours']==1/3600
    assert 'n_of_m' in report['sessions'][0]['ablations']
