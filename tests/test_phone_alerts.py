import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from browser_client import CameraClient as TestClient

from childsafe.notify import PhoneNotifier
from childsafe.web.app import create_app
from test_web_runtime import image, runtime

SUB = {'endpoint': 'https://push.example/abc', 'keys': {'p256dh': 'k', 'auth': 'a'}}


class FakeService(BaseHTTPRequestHandler):
    """Stands in for services/notify: in-memory subscriptions, records alerts."""
    subs, alerts, auth = [], [], []

    def _reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        FakeService.auth.append(self.headers.get('Authorization'))
        self._reply({'subscriptions': FakeService.subs})

    def do_POST(self):
        FakeService.auth.append(self.headers.get('Authorization'))
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/api/restore':
            FakeService.subs[:] = body['subscriptions']
            self._reply({'subscribers': len(FakeService.subs)})
        else:
            FakeService.alerts.append(body)
            self._reply({'sent': len(FakeService.subs), 'subscribers': len(FakeService.subs)})

    def log_message(self, *args):
        pass


@pytest.fixture
def service():
    FakeService.subs, FakeService.alerts, FakeService.auth = [], [], []
    server = ThreadingHTTPServer(('127.0.0.1', 0), FakeService)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_address[1]}'
    server.shutdown()


def wait(condition, timeout=5.):
    end = time.time() + timeout
    while time.time() < end and not condition():
        time.sleep(.02)
    return condition()


def test_alert_is_sent_and_lost_subscriptions_are_restored(service, tmp_path):
    store = tmp_path / 'subs.json'
    store.write_text(json.dumps([SUB]))     # saved earlier; the service has restarted empty
    notifier = PhoneNotifier(service, 'secret', store=store, keepalive_s=60)
    try:
        assert wait(lambda: FakeService.subs == [SUB])          # restored on start
        notifier.send(2, 'ChildSafe: NGUY HIỂM', 'Đang cầm dao')
        assert wait(lambda: FakeService.alerts and notifier.status['sent'] == 1)
        assert FakeService.alerts[0] == {'level': 2, 'title': 'ChildSafe: NGUY HIỂM', 'body': 'Đang cầm dao'}
        assert set(FakeService.auth) == {'Bearer secret'}
    finally:
        notifier.close()


def test_unreachable_service_is_reported_not_raised(tmp_path):
    notifier = PhoneNotifier('http://127.0.0.1:9', 'secret', store=tmp_path / 's.json', keepalive_s=60, timeout_s=1)
    try:
        notifier.send(2, 't', 'b')
        assert wait(lambda: notifier.status['last_error'])
    finally:
        notifier.close()
    with pytest.raises(ValueError):
        PhoneNotifier('http://example.com', 'secret')


def test_wrong_token_is_named_in_the_status(tmp_path):
    class Refuse(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(401); self.send_header('Content-Length', '0'); self.end_headers()

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Refuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    notifier = PhoneNotifier(f'http://127.0.0.1:{server.server_address[1]}', 'wrong', store=tmp_path / 's.json', keepalive_s=60)
    try:
        assert wait(lambda: notifier.status['last_error'].startswith('HTTP 401'))
    finally:
        notifier.close()
        server.shutdown()


def test_quiet_time_per_level_and_no_caution_right_after_danger(tmp_path):
    notifier = PhoneNotifier('http://127.0.0.1:9', 'secret', store=tmp_path / 's.json', keepalive_s=60, timeout_s=1)
    sent = []
    notifier.send = lambda *a: sent.append(a[0])
    try:
        assert notifier.alert(2, 'd', 'b', t=0) and not notifier.alert(2, 'd', 'b', t=10)   # same knife, new track ID
        assert not notifier.alert(1, 'c', 'b', t=30)            # caution right after danger
        assert notifier.alert(2, 'd', 'b', t=21) and notifier.alert(1, 'c', 'b', t=90)
        assert not notifier.alert(1, 'c', 'b', t=120)
        assert sent == [2, 2, 1]
    finally:
        notifier.close()


def test_web_demo_alerts_on_each_rise_without_images(monkeypatch):
    now = [0.]
    monkeypatch.setattr('childsafe.web.app.time', SimpleNamespace(perf_counter=lambda: now[0], strftime=time.strftime))
    monkeypatch.setattr('childsafe.pipeline.time', SimpleNamespace(perf_counter=lambda: now[0]))
    sent = []
    demo = runtime('unsafe')
    demo.notifier = SimpleNamespace(alert=lambda *a: sent.append(a), status={'last_error': ''}, url='https://x', close=lambda: None)
    client = TestClient(create_app(demo))
    try:
        for index in range(15):
            now[0] = index * .1
            client.post('/api/frame', content=image())
    finally:
        demo.close()
    levels = [a[0] for a in sent]
    assert levels and levels[-1] == 2 and levels == sorted(set(levels))   # one alert per rise, no repeats
    assert 'NGUY HIỂM' in sent[-1][1] and all(isinstance(a[2], str) for a in sent)
