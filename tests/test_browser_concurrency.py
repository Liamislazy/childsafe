import asyncio
import threading
import time
import httpx
import pytest
from childsafe.config import load_config
from childsafe.web.app import RuntimeDemo,create_app


def test_stop_during_inference_discards_reply_and_reset_does_not_queue():
    entered,release = threading.Event(),threading.Event()
    class Stub(RuntimeDemo):
        def __init__(self):
            self.cfg=load_config(); self.paths={}; self.lock=threading.Lock()
        def reset(self): pass
        def frame(self,*args,**kwargs):
            entered.set(); release.wait(3)
            return {'level':2,'green':False}
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(Stub())),base_url='http://test') as c:
            now=time.perf_counter()
            token=(await c.post('/api/camera/start',params={'client_t':now})).json()['token']
            assert (await c.post('/api/camera/sync',params={'token':token,'client_t':time.perf_counter()})).status_code==200
            pending=asyncio.create_task(c.post('/api/frame',params={'token':token,'sequence':1,'capture_t':time.perf_counter()},content=b'frame'))
            try:
                assert await asyncio.to_thread(entered.wait,2)
                assert (await c.post('/api/reset',params={'token':token})).status_code==503
                assert (await c.post('/api/camera/end',params={'token':token})).status_code==200
            finally:
                release.set()
            assert (await pending).status_code==409
    asyncio.run(scenario())


@pytest.mark.parametrize('stage',['decode','inference'])
def test_stop_does_not_commit_rows_or_risk_from_inflight_work(monkeypatch,stage):
    from test_web_runtime import runtime,image
    from childsafe.web import app as web
    demo=runtime('unsafe'); demo.t0=time.perf_counter()
    entered,release=threading.Event(),threading.Event()
    if stage=='decode':
        original=web._decode
        def blocked(*args,**kwargs):
            entered.set(); assert release.wait(3)
            return original(*args,**kwargs)
        monkeypatch.setattr(web,'_decode',blocked)
    else:
        original=demo.pipeline.process
        def blocked(*args,**kwargs):
            result=original(*args,**kwargs)
            entered.set(); assert release.wait(3)
            return result
        monkeypatch.setattr(demo.pipeline,'process',blocked)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(demo)),base_url='http://test') as c:
            token=(await c.post('/api/camera/start',params={'client_t':time.perf_counter()})).json()['token']
            await c.post('/api/camera/sync',params={'token':token,'client_t':time.perf_counter()})
            pending=asyncio.create_task(c.post('/api/frame',params={'token':token,'sequence':1,'capture_t':time.perf_counter()},content=image()))
            try:
                assert await asyncio.to_thread(entered.wait,2)
                assert (await c.post('/api/camera/end',params={'token':token})).status_code==200
            finally: release.set()
            assert (await pending).status_code==409
            assert demo.rows==[] and not demo.pipeline.states
            assert (await c.get('/api/session.json')).json()['end_s'] is not None
            newer=(await c.post('/api/camera/start',params={'client_t':time.perf_counter()})).json()['token']
            assert newer!=token and demo.rows==[]
    try: asyncio.run(scenario())
    finally: release.set(); demo.close()
