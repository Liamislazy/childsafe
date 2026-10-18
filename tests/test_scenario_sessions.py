import json
import pytest
from childsafe.evaluation.sessions import create_session, validate_sessions


def test_consent_and_subject_split_enforced_and_hashes_recorded(tmp_path):
    ckpt = tmp_path/'model'; ckpt.write_bytes(b'model')
    cfg = {'detector':{'weights':str(ckpt)}}
    with pytest.raises(ValueError): create_session(tmp_path,'a','dev',['p1'],'high',cfg,ckpt)
    directory = create_session(tmp_path,'a','dev',['p1'],'high',cfg,ckpt,consent_confirmed=True)
    metadata = json.loads((directory/'manifest.json').read_text())
    assert len(metadata['config_sha256']) == 64 and len(metadata['classifier_sha256']) == 64
    assert (directory/'annotations.csv').read_text() == 'start,end,label\n'
    with pytest.raises(ValueError):
        create_session(tmp_path,'b','test',['p1'],'high',cfg,ckpt,consent_confirmed=True)
    with pytest.raises(ValueError): validate_sessions([dict(metadata,subject_kind='child')])
