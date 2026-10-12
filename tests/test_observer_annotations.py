from fastapi.testclient import TestClient
from test_web_runtime import runtime


def test_observer_markers_are_separate_from_inference_and_bound_to_owner():
    from childsafe.web.app import create_app
    import time
    demo=runtime('unsafe'); c=TestClient(create_app(demo))
    try:
        token=c.post('/api/camera/start',params={'client_t':time.perf_counter()}).json()['token']
        assert c.post('/api/annotation',params={'token':'other','action':'start'}).status_code==409
        assert c.post('/api/annotation',params={'token':token,'action':'start','label':'non_event'}).status_code==200
        assert demo.rows==[]  # annotations do not fabricate model observations
        assert c.post('/api/annotation',params={'token':token,'action':'start'}).status_code==409
        assert c.post('/api/annotation',params={'token':token,'action':'end'}).status_code==200
        assert len(demo.annotations)==1 and 'non_event' in c.get('/api/annotations.csv').text
        assert c.post('/api/camera/end',params={'token':token}).status_code==200
        assert c.get('/api/session.json').json()['end_s'] is not None
    finally: demo.close()
