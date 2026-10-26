import numpy as np
from childsafe.vision.detection import SegmentationDetector


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()
    def item(self): return self.value.item()
    def __getitem__(self,i): return Tensor(self.value[i])


def test_adapter_bgr_conversion_and_full_resolution_mask():
    from types import SimpleNamespace
    rgb = np.zeros((12,20,3),np.uint8); rgb[:,:,0] = 200
    box = SimpleNamespace(xyxy=Tensor([[0,0,20,12]]),cls=Tensor(0),conf=Tensor(.9))
    result = SimpleNamespace(boxes=[box],names={0:'knife'},masks=SimpleNamespace(data=Tensor(np.ones((1,12,20)))))
    class Model:
        def predict(self,image,**kwargs):
            assert image[0,0].tolist() == [0,0,200]
            assert kwargs['retina_masks'] and not kwargs.get('save',False)
            return [result]
    adapter = SegmentationDetector.__new__(SegmentationDetector)
    adapter.model,adapter.confidence,adapter.error = Model(),.5,''
    detections = adapter.detect(rgb)
    assert detections[0].mask.shape == rgb.shape[:2]
    result.masks.data = Tensor(np.ones((1,6,10)))
    assert adapter.detect(rgb)[0].mask is None


def test_coco_toothbrush_is_shown_as_pen_hint():
    from types import SimpleNamespace
    box = SimpleNamespace(xyxy=Tensor([[0,0,20,12]]),cls=Tensor(0),conf=Tensor(.9))
    result = SimpleNamespace(boxes=[box],names={0:'toothbrush'},masks=None)
    class Model:
        def predict(self,image,**kwargs): return [result]
    adapter = SegmentationDetector.__new__(SegmentationDetector)
    adapter.model,adapter.confidence,adapter.error = Model(),.5,''
    assert [d.label for d in adapter.detect(np.zeros((12,20,3),np.uint8))] == ['pen']


def test_inference_failure_returns_explicit_limited_mode():
    class Broken:
        def predict(self,*a,**kw): raise RuntimeError('private source text')
    adapter = SegmentationDetector.__new__(SegmentationDetector)
    adapter.model,adapter.confidence,adapter.error = Broken(),.5,''
    assert adapter.detect(np.zeros((12,20,3),np.uint8)) == []
    assert not adapter.ok and adapter.error == 'RuntimeError'
