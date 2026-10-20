import json
import pandas as pd
import pytest

from childsafe.config import load_config
from childsafe.evaluation.fair_evidence import collect_evidence, write_evidence
from childsafe.evaluation.sessions import create_session


def test_no_sessions_produces_empty_table_not_zero_false_alarms(tmp_path):
    bundle = collect_evidence(tmp_path/'absent')
    out = write_evidence(bundle, tmp_path/'out')
    assert bundle['status'] == 'not_measured' and bundle['report'] is None
    assert len(pd.read_csv(out/'comparison.csv')) == 0
    assert 'Chưa đo' in (out/'summary.md').read_text(encoding='utf8')
    assert json.loads((out/'evidence.json').read_text(encoding='utf8'))['comparison'] == []


def test_incomplete_sessions_fail_instead_of_becoming_results(tmp_path):
    model = tmp_path/'model.pt'; model.write_bytes(b'fixture')
    cfg = load_config(); cfg['detector']['weights'] = str(model)
    create_session(tmp_path/'sessions', 'test1', 'test', ['adult1'], 'front',
                   cfg, model, consent_confirmed=True)
    with pytest.raises(FileNotFoundError):
        collect_evidence(tmp_path/'sessions')


def test_labeled_negative_session_keeps_unknown_recall_and_separate_baseline(tmp_path):
    model = tmp_path/'model.pt'; model.write_bytes(b'fixture')
    cfg = load_config(); cfg['detector']['weights'] = str(model)
    folder = create_session(tmp_path/'sessions', 'test1', 'test', ['adult1'], 'front',
                            cfg, model, consent_confirmed=True)
    path = folder/'manifest.json'
    m = json.loads(path.read_text()); m.update(start_s=0., end_s=1.)
    path.write_text(json.dumps(m))
    provenance = {k:m[k] for k in ('config_sha256', 'classifier_sha256', 'detector_sha256', 'code_sha256')}
    pd.DataFrame([dict(t=t,session_id=1,person_id=0,schema_version=2,mouth_visible=False,
                       face_ok=False,hand_ok=False,d_face_width=1.,**provenance) for t in (0,.1,.2)]
                 ).to_csv(folder/'features.csv', index=False)
    pd.DataFrame([dict(start=0.,end=1.,label='non_event')]).to_csv(folder/'annotations.csv', index=False)
    bundle = collect_evidence(tmp_path/'sessions')
    assert bundle['status'] == 'partial_labeled_sessions'
    assert {r['policy'] for r in bundle['comparison']} == {'full_danger','distance_only','full_caution_or_danger'}
    assert all(r['event_recall'] is None for r in bundle['comparison'])
    assert all(r['negative_hours'] == 1/3600 for r in bundle['comparison'])
    assert all(r['image_observation_coverage'] == 0 for r in bundle['comparison'])
