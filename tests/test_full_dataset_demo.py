"""Opt-in integration of the downloaded checkpoint with the real demo API."""
import io
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from childsafe.classifier.calibration import softmax
from childsafe.config import load_config
from childsafe.data.dataset import Letterbox
from childsafe.web.app import Demo, _prediction_status, create_app


@pytest.mark.skipif(not os.environ.get('CHILDSAFE_FULL_DATASET_CKPT'), reason='requires local full-data checkpoint and MediaPipe models')
def test_downloaded_model_demo_preprocessing_calibration_and_unknown_rejection(monkeypatch):
    checkpoint = Path(os.environ['CHILDSAFE_FULL_DATASET_CKPT'])
    cfg = load_config()
    demo = Demo(cfg, {'full dataset (Kaggle)': str(checkpoint)})
    try:
        clf = demo.classifiers['full dataset (Kaggle)']
        assert len(clf.classes) == 11 and 'other' in clf.classes
        assert clf.temperature == pytest.approx(1.7798281300110244)
        assert isinstance(clf.tf.transforms[0], Letterbox)
        rgb = np.full((80, 320, 3), 140, dtype=np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format='PNG')
        client = TestClient(create_app(demo))
        result = clf(rgb)                      # the photo-test endpoint was removed; check the classifier
        assert len(result['top3']) == 3
        with torch.inference_mode():
            logits = clf.model(clf.tf(Image.fromarray(rgb)).unsqueeze(0).to(clf.device)).cpu().numpy()
        expected = softmax(logits, clf.temperature)[0][clf.unsafe].sum()
        assert result['p_unsafe_cal'] == pytest.approx(float(expected))
        # Exercise actual checkpoint class ordering/binary map, where other is
        # encoded safe for training but must remain uncertain in the demo.
        def force_other(x):
            out = torch.full((len(x), 11), -20., device=x.device)
            out[:, clf.classes.index('other')] = 20.
            return out
        monkeypatch.setattr(clf.model, 'forward', force_other)
        photo = clf(rgb)
        assert _prediction_status(photo, cfg['calibration']['reject_threshold']) == 'uncertain'
        assert photo['top3'][0][0] == 'other'
        frame = client.post('/api/frame?mode=full', content=encoded.getvalue())
        assert frame.status_code == 200
        live = frame.json()['models']['full dataset (Kaggle)']
        assert live['obj_status'] == 'uncertain' and not live['green']
        assert live['observation'] == 'unobservable'
    finally:
        demo.detector.close()


@pytest.mark.skipif(not os.environ.get('CHILDSAFE_FULL_DATASET_CKPT'), reason='requires local full-data checkpoint and detector weights')
def test_real_runtime_web_api_and_feature_export():
    from browser_client import CameraClient as TestClient
    from childsafe.web.app import RuntimeDemo
    demo = RuntimeDemo(load_config(), {'full dataset':os.environ['CHILDSAFE_FULL_DATASET_CKPT']})
    try:
        image = io.BytesIO()
        Image.fromarray(np.full((240,320,3),140,np.uint8)).save(image,format='PNG')
        client = TestClient(create_app(demo))
        response = client.post('/api/frame',content=image.getvalue())
        assert response.status_code == 200
        out = response.json()
        assert out['detector_ok'] and out['classifier_ok']
        assert out['people'] == [] and not out['green'] and out['level'] == 0
        assert out['observation'] == 'unobservable'
        assert 'session_id' in client.get('/api/session.csv').text
        assert client.post('/api/reset').status_code == 200
    finally:
        demo.close()
