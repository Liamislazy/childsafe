from fastapi.testclient import TestClient
from childsafe.web import app as web


class CameraClient(TestClient):
    """Exercise the actual camera clock/session handshake in existing integration cases."""
    def post(self, url, **kwargs):
        if url.startswith(('/api/frame','/api/reset')):
            if not hasattr(self,'camera_token'):
                now = web.time.perf_counter()
                started = super().post('/api/camera/start',params={'client_t':now})
                assert started.status_code == 200
                self.camera_token = started.json()['token']
                assert super().post('/api/camera/sync',params={'token':self.camera_token,'client_t':now}).status_code == 200
                self.sequence = 0
            params = dict(kwargs.pop('params',{}),token=self.camera_token)
            if url.startswith('/api/frame'):
                self.sequence += 1
                params.update(sequence=self.sequence,capture_t=web.time.perf_counter())
            kwargs['params'] = params
        return super().post(url,**kwargs)
