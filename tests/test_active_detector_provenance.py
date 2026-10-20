"""Selected model identity must agree across live export and prepared sessions."""
import hashlib
import json
from types import SimpleNamespace

import torch

from childsafe.config import load_config
from childsafe.evaluation.sessions import create_session
from childsafe.vision.detection import describe_selection


def test_gpu_session_records_selected_weights_not_cpu_default(tmp_path, monkeypatch):
    base, gpu, ckpt = [tmp_path / name for name in ('n.pt', 's.pt', 'clf.pt')]
    for path in (base, gpu, ckpt):
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    cfg = load_config()
    cfg['detector'].update(weights=str(base), weights_gpu=str(gpu), device='auto')
    directory = create_session(tmp_path/'sessions', 'dev01', 'dev', ['adult1'],
                               'elevated', cfg, ckpt, consent_confirmed=True)
    manifest = json.loads((directory/'manifest.json').read_text())
    assert manifest['detector_sha256'] == hashlib.sha256(gpu.read_bytes()).hexdigest()
    assert describe_selection(cfg['detector'])['file'] == 's.pt'
    gpu.unlink()
    selection = describe_selection(cfg['detector'])
    assert selection['fallback'] and selection['file'] == 'n.pt'


def test_web_keeps_pipeline_selected_hash(tmp_path, monkeypatch):
    import childsafe.web.app as web
    ckpt = tmp_path/'classifier.pt'
    ckpt.write_bytes(b'local classifier fixture')
    chosen_hash = 'b'*64
    monkeypatch.setattr(web, 'ObjectClassifier', lambda path: object())
    monkeypatch.setattr(web, 'FramePipeline', lambda *a, **kw: SimpleNamespace(
        provenance={'detector_sha256': chosen_hash, 'config_sha256':'c'*64, 'code_sha256':'d'*64},
        detector_selection={'file':'s.pt','fallback':False}))
    demo = web.RuntimeDemo(load_config(), {'fixture':str(ckpt)})
    assert demo.provenance['detector_sha256'] == chosen_hash
    assert demo.pipeline.provenance == demo.provenance
    assert demo.provenance['classifier_sha256'] == hashlib.sha256(ckpt.read_bytes()).hexdigest()


def test_explicit_detector_checksum_overrides_automatic_configuration():
    from childsafe.pipeline import FramePipeline
    actual = SimpleNamespace(checksum='e'*64, weights_file='explicit.pt', device='cpu')
    pipeline = FramePipeline(load_config(), None, detector=actual)
    assert pipeline.provenance['detector_sha256'] == 'e'*64
    assert pipeline.detector_selection['file'] == 'explicit.pt'
    assert pipeline.detector_selection['selection_reason'] == 'explicit_detector'
