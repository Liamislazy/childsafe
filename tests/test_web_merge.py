import io
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from childsafe.config import load_config
from childsafe.web.app import Demo, create_app


def demo():
    app_demo = Demo.__new__(Demo)
    app_demo.cfg,app_demo.lock = load_config(),threading.Lock()
    app_demo.t0 = 0.
    def classifier(*args):
        return dict(p_unsafe_raw=.01,p_unsafe_cal=.01,obj_conf=.99,obj_class='other',p_unknown_cal=.99,
                    top3=[('other',.99),('knife',.01)])
    app_demo.classifiers = {'comparison':classifier}
    app_demo.paths = {'comparison':'unused/best.pt'}
    class Detector:
        def detect(self,rgb,t):
            return SimpleNamespace(t=t,width=rgb.shape[1],height=rgb.shape[0],face=None,hands=[],face_matrix=None)
    app_demo.detector = Detector()
    app_demo.reset()
    return app_demo


def encoded_image():
    stream = io.BytesIO(); Image.fromarray(np.zeros((20,40,3),np.uint8)).save(stream,format='PNG')
    return stream.getvalue()


def test_photo_demo_does_not_label_other_safe():
    client = TestClient(create_app(demo()))
    response = client.post('/api/image',content=encoded_image())
    assert response.status_code == 200
    assert response.json()['models']['comparison']['status'] == 'uncertain'
    assert response.json()['models']['comparison']['top3'][0][0] == 'other'


def test_live_demo_has_limited_observation_without_contact_detector():
    client = TestClient(create_app(demo()))
    response = client.post('/api/frame?mode=full',content=encoded_image())
    assert response.status_code == 200
    out = response.json()['models']['comparison']
    assert not out['green'] and out['observation'] == 'unobservable'
    assert out['obj_status'] == 'uncertain'
    assert out['level'] != 2
    assert client.post('/api/reset').status_code == 200
    assert client.get('/').status_code == 200
