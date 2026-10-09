"""Opt-in integration with approved local images; default runs need no model downloads."""
import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.mark.skipif(not os.environ.get('CHILDSAFE_TEST_IMAGE'),reason='requires approved local object image and local weights')
def test_real_segmentation_classifier_and_landmark_models():
    from childsafe.vision.detection import SegmentationDetector
    from childsafe.vision.object_classifier import ObjectClassifier
    from childsafe.vision.landmarks import LandmarkDetector
    from childsafe.config import load_config
    from childsafe.pipeline import FramePipeline
    from childsafe.evaluation.replay import replay
    import pandas as pd
    rgb = np.asarray(Image.open(os.environ['CHILDSAFE_TEST_IMAGE']).convert('RGB'))
    detector = SegmentationDetector(os.environ['CHILDSAFE_TEST_DETECTOR'])
    detections = detector.detect(rgb)
    assert detector.ok
    assert any(d.mask is not None and d.mask.shape == rgb.shape[:2] and d.mask.any() for d in detections)
    classifier = ObjectClassifier(os.environ['CHILDSAFE_TEST_CKPT'],device='cpu')
    obj = classifier(rgb)
    assert 0 <= obj['obj_conf'] <= 1 and 0 <= obj['p_unsafe_cal'] <= 1
    with LandmarkDetector(os.environ['CHILDSAFE_TEST_FACE'],os.environ['CHILDSAFE_TEST_HAND']) as lm:
        result = lm.detect(rgb,0)
        assert result.width == rgb.shape[1] and result.height == rgb.shape[0]
        assert result.face is None  # this approved test crop contains only an object
    cfg = load_config()
    pipe = FramePipeline(cfg,None,detector=detector,classifier=classifier)
    try:
        _,_,out = pipe.process(rgb,0)
        assert not out['green']
        offline = replay(pd.DataFrame(out['feature_rows']),cfg)
        assert offline.level.max() == out['level']
        assert (offline.observation == 'unobservable').all()
    finally:
        pipe.close()
