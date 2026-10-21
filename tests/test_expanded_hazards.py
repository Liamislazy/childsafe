"""Regression for real camera failures: unknown crop, new classes and model conflicts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from childsafe.alerts.processor import AlertProcessor
from childsafe.config import load_config
from childsafe.evaluation.replay import replay
from childsafe.product import product_scope
from childsafe.vision.detection import Detection, SupplementalDetector, detector_hash
from test_observation import f


def run(category='scissors', score=.9, crop='other', mask=True, **features):
    cfg = load_config(); proc = AlertProcessor(cfg); rows = []; outputs = []
    for i in range(20):
        row = f(i*.1)
        row.update(object_present=True, object_label=category, object_conf=score,
                   mask_reliable=mask, mask_temporally_stable=mask, proximity_evidence=mask)
        row.update(features)
        obj = dict(obj_class=crop, obj_conf=.99, p_unknown_cal=.99 if crop == 'other' else 0.,
                   p_unsafe_raw=.99 if crop == 'knife' else .01,
                   p_unsafe_cal=.99 if crop == 'knife' else .01)
        outputs.append(proc.step(row, obj)); rows.append(dict(row, **obj))
    assert replay(pd.DataFrame(rows), cfg).level.tolist() == [o['level'] for o in outputs]
    return outputs[-1]


@pytest.mark.parametrize('category', ['knife','fork','scissors','pen','pencil','chopsticks',
    'coin','button_battery','cylindrical_battery','screw','nail','safety_pin','needle','marble','small_toy_piece'])
def test_real_trained_detector_category_can_warn_without_old_classifier_class(category):
    # Synthetic category evidence verifies rules, not model recognition/accuracy.
    out = run(category)
    assert out['level'] == 2 and out['hazard_source'] == 'detector'
    assert out['obj_class'] == 'other' and out['obj_status'] == 'uncertain'


@pytest.mark.parametrize('kwargs', [dict(score=.5), dict(mask=False), dict(detector_ok=False),
    dict(mouth_visible=False), dict(association_ambiguous=True), dict(frame_valid=False),
    dict(category='cell phone'), dict(crop='fruit')])
def test_unreliable_conflicting_or_unrelated_evidence_never_upgrades_to_danger(kwargs):
    assert run(**kwargs)['level'] <= 1


def test_scope_does_not_claim_new_classes_before_loading_weights():
    cfg = load_config()
    assert 'pen' in product_scope(cfg)['pending_training_categories']
    assert product_scope(cfg)['additional_detector_categories'] == []
    detector = SimpleNamespace(names=['person','knife','pen'], ok=True)
    scope = product_scope(cfg, detector)
    assert scope['additional_detector_categories'] == ['pen']
    assert 'pen' not in scope['pending_training_categories']
    detector.ok = False
    assert product_scope(cfg, detector)['additional_detector_categories'] == []


def test_supplemental_preserves_person_model_and_never_invents_mask():
    class Model:
        ok = True; error = ''; weights_file = 'fixture.pt'; device = 'cpu'
        def __init__(self, items): self.items = items; self.names = [d.label for d in items]
        def detect(self, rgb): return self.items
    person = Detection((0,0,100,200), 'person', .99)
    phone = Detection((10,10,50,50), 'cell phone', .95)
    pen = Detection((10,10,50,50), 'pen', .8)
    detector = SupplementalDetector(Model([person,phone]), Model([pen]), 'fixture')
    rgb = np.zeros((200,100,3),np.uint8)
    assert detector.detect(rgb) == [person,phone]
    found = detector.detect_roi(rgb)
    assert [d.label for d in found] == ['person','pen']
    assert found[1].mask is None and found[1].source == 'supplemental'


def test_detector_provenance_changes_when_supplemental_weights_change(tmp_path):
    base, extra = tmp_path/'base.pt', tmp_path/'objects.pt'
    base.write_bytes(b'base'); extra.write_bytes(b'one')
    cfg = dict(weights=str(base), device='cpu')
    original = detector_hash(cfg)
    cfg['supplemental_weights'] = str(extra)
    combined = detector_hash(cfg)
    extra.write_bytes(b'two')
    assert len(combined) == 64 and combined != original and detector_hash(cfg) != combined


def test_supplemental_pipeline_box_can_request_caution_but_never_mask_confirmation():
    from childsafe.pipeline import FramePipeline
    from test_runtime_optimization import Face, RGB, configuration
    class Model:
        ok=True; error=''; weights_file='fixture.pt'; device='cpu'; names=['person','pen']
        def __init__(self,person): self.person=person
        def detect(self,rgb):
            h,w=rgb.shape[:2]
            return [Detection((0,0,w,h),'person',.99)] if self.person else [Detection((w*.4,h*.4,w*.65,h*.65),'pen',.9)]
    cfg=configuration()
    detector=SupplementalDetector(Model(True),Model(False),'fixture')
    pipe=FramePipeline(cfg,None,detector=detector,landmark_factory=Face,
                       classifier=lambda *args:dict(obj_class='other',obj_conf=.99,p_unsafe_raw=.01,p_unsafe_cal=.01,p_unknown_cal=.99))
    try:
        for i in range(20):
            _,row,out=pipe.process(RGB,i*.1)
            assert out['level']<=1 and not row['proximity_evidence']
        assert row['object_detector_source']=='supplemental' and row['object_label']=='pen'
        assert row['detector_calls']==3 and out['level']==1
    finally: pipe.close()
