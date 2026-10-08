import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from childsafe.classifier.evaluate import _cached_images, _CACHE
from childsafe.classifier.train import class_balanced_sampler
from childsafe.data.dataset import image_paths, CatalogDataset, eval_transform


def test_absolute_supplement_paths_coexist_with_source_roots(tmp_path):
    absolute = tmp_path/'reviewed.jpg'
    df = pd.DataFrame(dict(path=['same.jpg','same.jpg',str(absolute)],source=['childsun','openimages',np.nan],
                           category=['fruit']*3))
    roots = dict(childsun=tmp_path/'cs',openimages=tmp_path/'oi')
    assert image_paths(df,roots) == [roots['childsun']/'same.jpg',roots['openimages']/'same.jpg',absolute]
    ds = CatalogDataset(df,roots,['fruit'],eval_transform(224,preprocessing='letterbox'))
    assert ds.paths[-1] == absolute


def test_evaluation_cache_preserves_source_and_preprocessing(tmp_path):
    roots = dict(childsun=tmp_path/'cs',openimages=tmp_path/'oi')
    for name,color in [('childsun',(255,0,0)),('openimages',(0,255,0))]:
        roots[name].mkdir(); Image.new('RGB',(200,20),color).save(roots[name]/'same.png')
    df = pd.DataFrame(dict(path=['same.png','same.png'],source=['childsun','openimages']))
    _CACHE.clear()
    letterbox = _cached_images(df,roots,224,'letterbox')
    center = _cached_images(df,roots,224,'center_crop')
    assert len(_CACHE) == 2
    assert np.asarray(letterbox[0])[112,112].tolist() == [255,0,0]
    assert np.asarray(letterbox[1])[112,112].tolist() == [0,255,0]
    assert np.asarray(letterbox[0])[0,0].tolist() != np.asarray(center[0])[0,0].tolist()


def test_upstream_sqrt_sampling_still_favors_rare_classes():
    sampler = class_balanced_sampler(torch.tensor([0,0,0,1]),2,'sqrt')
    assert sampler.replacement and sampler.num_samples == 4
    assert sampler.weights[-1] > sampler.weights[0]
    assert class_balanced_sampler(torch.tensor([0,1]),2,None) is None


def test_classifier_preserves_top3_unknown_mass_and_checkpoint_preprocessing(monkeypatch):
    import childsafe.vision.object_classifier as module
    class Fixed(torch.nn.Module):
        def forward(self,x):
            assert tuple(x.shape) == (1,3,224,224)
            return torch.tensor([[.05,.9,.05]]).log()
    ck = dict(classes=['knife','other','fruit'],temperature=1.,img_size=224,preprocessing='letterbox',
              binary_map={'knife':'unsafe','other':'safe','fruit':'safe'})
    monkeypatch.setattr(module,'load_checkpoint',lambda *a:(Fixed(),ck))
    classifier = module.ObjectClassifier('unused.pt',device='cpu')
    out = classifier(np.zeros((20,200,3),np.uint8))
    assert out['top3'][0][0] == 'other'
    assert out['p_unknown_cal'] > .8
    assert classifier.tf.transforms[0].__class__.__name__ == 'Letterbox'


def test_ood_status_rejects_other_and_low_category_confidence():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('compare_ood',root/'scripts/compare_ood.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    status = module.status(np.array([.05,.05,.9]),.7,np.array([.99,.3,.9]),
                           np.array(['other','fruit','knife']),np.array([.99,0.,0.]))
    assert status.tolist() == ['uncertain','uncertain','unsafe']


def test_rejection_metrics_exclude_confident_other_predictions():
    from childsafe.classifier.engine import summarize
    result = summarize(np.log([[.01,.99]]),np.array([1]),['knife','other'],
                       {'knife':'unsafe','other':'safe'},reject_threshold=.7)
    assert result['accuracy'] == 1.
    assert result['reject']['coverage'] == 0.


def test_gradcam_letterbox_keeps_overlay_aligned_with_model_input(tmp_path, monkeypatch):
    import childsafe.classifier.gradcam as module
    images = tmp_path/'images'; images.mkdir()
    run = tmp_path/'run'; run.mkdir()
    Image.new('RGB',(200,20),(255,255,255)).save(images/'thin.png')
    ck = dict(arch='unused',img_size=224,preprocessing='letterbox',temperature=1.,classes=['knife'])
    monkeypatch.setattr(module,'load_checkpoint',lambda *args:(None,ck))
    monkeypatch.setattr(module,'gradcam_target_layer',lambda *args:None)
    class FakeCAM:
        def __init__(self,*args): pass
        def __call__(self,x):
            assert tuple(x.shape) == (1,3,224,224)
            assert not torch.equal(x[0,:,0,0],x[0,:,112,112])
            return np.zeros((224,224)),np.array([1.]),0
    monkeypatch.setattr(module,'GradCAM',FakeCAM)
    module.main(['--run',str(run),'--images',str(images)])
    result = np.asarray(Image.open(run/'gradcam/000_thin.jpg'))
    assert result.shape == (224,224,3)
    assert result[112,112].mean() > result[210,112].mean() + 40
