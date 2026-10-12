from types import SimpleNamespace
import numpy as np
from childsafe.vision.elevated import ElevatedQuality


def landmarks(scale=1):
    face = np.tile([40., 40., 0.], (478, 1))
    face[[78, 13, 308, 14], :2] = [[40,45], [45,40], [50,45], [45,50]]
    return SimpleNamespace(face=face, face_matrix=np.eye(4), hands=[], t=0., scale=scale)


def test_zoom_does_not_make_small_original_face_observable():
    rgb = np.random.default_rng(3).integers(40, 210, (100,100,3), dtype=np.uint8)
    results = [ElevatedQuality({}).measure(rgb, landmarks(s), {'face_width':20}) for s in (1,2,3)]
    assert all(not r['mouth_visible'] for r in results)
    assert all(r['original_face_px'] == 20 for r in results)
    assert all('face too small' in r['quality_reasons'] for r in results)


def test_face_roi_dark_or_blurred_despite_bright_textured_background():
    rgb = np.random.default_rng(4).integers(60, 220, (100,100,3), dtype=np.uint8)
    rgb[40:51,40:51] = 10
    r = ElevatedQuality({}).measure(rgb, landmarks(), {'face_width':60})
    assert 'poor exposure' in r['quality_reasons'] and not r['mouth_visible']
    rgb[40:51,40:51] = 130
    r = ElevatedQuality({}).measure(rgb, landmarks(), {'face_width':60})
    assert 'blurred image' in r['quality_reasons'] and not r['mouth_visible']
    assert r['quality_roi_x0'] == 40 and r['quality_roi_x1'] == 51


def test_clipped_empty_face_roi_cannot_be_observable():
    lm = landmarks(); lm.face[:, :2] += 200
    r = ElevatedQuality({}).measure(np.full((100,100,3),120,np.uint8), lm, {'face_width':60})
    assert not r['mouth_visible'] and r['brightness'] == 0
