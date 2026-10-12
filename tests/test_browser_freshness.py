import pytest
from childsafe.runtime.browser import BrowserLease, FrameRejected


def lease():
    now = [100.]
    gate = BrowserLease(clock=lambda:now[0])
    token = gate.start(10.)['token']
    gate.sync(token,10.02)
    return gate, token, now


def test_clock_offset_uncertainty_duplicates_future_and_stale():
    g, token, now = lease()
    deadline, age = g.frame(token,1,10.02)
    assert deadline == pytest.approx(100.25) and age == pytest.approx(0,abs=1e-12)
    for seq,t in [(1,10.03),(2,10.02),(2,11.),(2,9.)]:
        with pytest.raises(FrameRejected): g.frame(token,seq,t)
    now[0] = 100.4
    with pytest.raises(FrameRejected): g.frame(token,2,10.03)


def test_runtime_api_requires_owner_and_rejects_duplicate_without_new_rows():
    from test_web_runtime import runtime, image
    from fastapi.testclient import TestClient
    import time
    demo = runtime('unsafe'); client = TestClient(__import__('childsafe.web.app',fromlist=['create_app']).create_app(demo))
    try:
        assert client.post('/api/frame',content=image()).status_code == 409
        now = time.perf_counter()
        token = client.post('/api/camera/start',params={'client_t':now}).json()['token']
        assert client.post('/api/camera/sync',params={'token':token,'client_t':time.perf_counter()}).status_code == 200
        assert client.post('/api/camera/start',params={'client_t':now}).status_code == 409
        args = dict(token=token,sequence=1,capture_t=time.perf_counter())
        assert client.post('/api/frame',params=args,content=image()).status_code == 200
        n = len(demo.rows)
        assert client.post('/api/frame',params=args,content=image()).status_code == 409
        assert len(demo.rows) == n
        assert client.post('/api/reset',params={'token':'other-tab'}).status_code == 409
        assert client.post('/api/camera/end',params={'token':token}).status_code == 200
        assert client.post('/api/frame',params=args,content=image()).status_code == 409
    finally: demo.close()


def test_owner_expiry_end_and_renew_keep_sequences():
    g, token, now = lease()
    with pytest.raises(FrameRejected): g.start(10.1)
    with pytest.raises(FrameRejected): g.frame('other-tab',1,10.1)
    g.frame(token,1,10.02)
    g.renew(token,10.02); g.sync(token,10.02)
    with pytest.raises(FrameRejected): g.frame(token,1,10.03)
    g.end(token)
    with pytest.raises(FrameRejected): g.check(token)
    newer = g.start(10.1)['token']; assert newer != token
    now[0] += 31
    with pytest.raises(FrameRejected): g.check(newer)


def test_unverified_clock_and_large_uncertainty_rejected():
    g = BrowserLease(clock=lambda:1.)
    token = g.start(0.)['token']
    with pytest.raises(FrameRejected): g.frame(token,1,0.)
    with pytest.raises(FrameRejected): g.sync(token,.3)
    with pytest.raises(FrameRejected): g.sync(token,float('nan'))


def test_pipeline_expired_processing_cannot_publish_new_danger():
    from test_web_runtime import runtime
    import numpy as np
    from childsafe.evaluation.replay import replay
    import pandas as pd
    demo = runtime('unsafe')
    try:
        _,_,out = demo.pipeline.process(np.full((160,160,3),140,np.uint8),0.,processing_deadline=-1.)
        assert not out['green'] and out['level'] == 0
        assert all(not f['frame_valid'] for f in out['feature_rows'])
        assert replay(pd.DataFrame(out['feature_rows']),demo.cfg).level.max() == 0
    finally: demo.close()
