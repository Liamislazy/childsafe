import io
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
from browser_client import CameraClient as TestClient

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import Detection
from childsafe.web.app import RuntimeDemo, create_app


class Landmarks:
    def detect(self, rgb, t):
        face = np.zeros((478,3))
        face[234,:2] = [20,40]; face[454,:2] = [100,40]
        face[[78,13,308,14],:2] = [[50,50],[60,45],[70,50],[60,55]]
        face[[33,133],:2] = [40,30]; face[[362,263],:2] = [80,30]
        hand = np.column_stack((np.linspace(50,70,21),np.linspace(50,65,21),np.zeros(21)))
        return SimpleNamespace(t=t,width=rgb.shape[1],height=rgb.shape[0],face=face,
                               hands=[hand],face_matrix=np.eye(4))
    def close(self): pass


def runtime(kind):
    class Detector:
        ok = True
        def detect(self, rgb):
            if kind == 'empty': return []
            result = [Detection((0,0,160,160),'person',.99)]
            if kind in ('unsafe','other'):
                mask = np.zeros(rgb.shape[:2],bool); mask[45:60,65:80] = True
                result.append(Detection((65,45,80,60),'knife',.99,mask))
            return result
    def classifier(*args):
        return dict(p_unsafe_raw=.99 if kind=='unsafe' else .01,
                    p_unsafe_cal=.99 if kind=='unsafe' else .01,
                    obj_conf=.99,obj_class='knife' if kind=='unsafe' else 'other',
                    p_unknown_cal=.99 if kind=='other' else 0.,top3=[])
    obj = RuntimeDemo.__new__(RuntimeDemo)
    obj.cfg = load_config(); obj.cfg['quality']['min_sharpness'] = 0
    obj.lock = threading.Lock(); obj.t0 = 0.
    obj.paths = {'full dataset':'unused/best.pt'}; obj.live_name = 'full dataset'
    obj.rows = []; obj.recording_full = False
    obj.classifiers = {'full dataset':classifier}
    obj.pipeline = FramePipeline(obj.cfg,None,detector=Detector(),landmark_factory=Landmarks,classifier=classifier)
    return obj


def image():
    out = io.BytesIO(); Image.fromarray(np.full((160,160,3),140,np.uint8)).save(out,format='PNG')
    return out.getvalue()


def test_web_runtime_delivers_danger_mask_person_and_notification(monkeypatch):
    now = [0.]
    monkeypatch.setattr('childsafe.web.app.time',SimpleNamespace(perf_counter=lambda:now[0]))
    monkeypatch.setattr('childsafe.pipeline.time',SimpleNamespace(perf_counter=lambda:now[0]))
    demo = runtime('unsafe'); client = TestClient(create_app(demo))
    try:
        states = []
        for index in range(15):
            now[0] = index*.1
            response = client.post('/api/frame',content=image())
            assert response.status_code == 200
            states.append(response.json())
        last = states[-1]
        assert last['level'] == 2 and last['people'][0]['level'] == 2
        assert last['people'][0]['mask_reliable'] and last['objects'][0]['mask']
        assert last['people'][0]['hands'] and last['people'][0]['mouth_visible']
        assert any(state['notify'] for state in states)
        assert not last['green']
        assert client.get('/api/info').json()['runtime']
        csv = client.get('/api/session.csv')
        assert csv.status_code == 200 and 'person_id' in csv.text and 'proximity_evidence' in csv.text
        assert client.post('/api/reset').status_code == 200
        assert demo.rows == []
        now[0] += .1
        assert client.post('/api/frame',content=image()).json()['level'] != 2
    finally: demo.close()


def test_web_runtime_accepts_independent_strong_detector_but_not_unlocalized_hand(monkeypatch):
    now = [0.]
    monkeypatch.setattr('childsafe.web.app.time',SimpleNamespace(perf_counter=lambda:now[0]))
    monkeypatch.setattr('childsafe.pipeline.time',SimpleNamespace(perf_counter=lambda:now[0]))
    for kind in ['other','hand_only','empty']:
        demo = runtime(kind); client = TestClient(create_app(demo))
        try:
            for index in range(15):
                now[0] = index*.1
                out = client.post('/api/frame',content=image()).json()
            assert out['level'] == {'other':2, 'hand_only':1, 'empty':0}[kind]
            assert not out['green']
            if kind=='other':
                assert out['people'][0]['status'] == 'uncertain'
                assert out['people'][0]['hazard_source'] == 'detector'
                assert out['people'][0]['object'] == 'knife'
            if kind=='hand_only': assert not out['people'][0]['mask_reliable']
            if kind=='empty': assert not out['people'] and out['limitations']
        finally: demo.close()
