"""Run with the service's own venv: .venv-notify/Scripts/python -m pytest services/notify/test_app.py"""
import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SPEC = importlib.util.spec_from_file_location('notify_app', Path(__file__).with_name('app.py'))
SUB = {'endpoint': 'https://push.example/abc', 'keys': {'p256dh': 'k', 'auth': 'a'}}
AUTH = {'Authorization': 'Bearer laptop-secret'}


@pytest.fixture
def service(monkeypatch):
    for key, value in dict(VAPID_PUBLIC_KEY='pub', VAPID_PRIVATE_KEY='priv', ALERT_TOKEN='laptop-secret',
                           MONITOR_CODE='abc123').items():
        monkeypatch.setenv(key, value)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    sent = []
    monkeypatch.setattr(module, 'webpush', lambda **kw: sent.append(kw))
    return module, TestClient(module.app), sent


def test_phone_subscribes_with_code_and_laptop_alert_is_pushed(service):
    module, client, sent = service
    assert client.post('/api/subscribe', json={'code': 'wrong', 'subscription': SUB}).status_code == 401
    assert client.post('/api/subscribe', json={'code': 'abc123', 'subscription': {'endpoint': 'http://x'}}).status_code == 422
    assert client.post('/api/subscribe', json={'code': 'abc123', 'subscription': SUB}).json() == {'subscribers': 1}
    alert = {'level': 2, 'title': 'ChildSafe: NGUY HIỂM', 'body': 'Đang cầm dao'}
    assert client.post('/api/alert', json=alert).status_code == 401
    assert client.post('/api/alert', json=alert, headers={'Authorization': 'Bearer nope'}).status_code == 401
    assert client.post('/api/alert', json=alert, headers=AUTH).json() == {'sent': 1, 'subscribers': 1}
    assert sent[0]['subscription_info'] == SUB and 'Đang cầm dao' in sent[0]['data']
    assert client.get('/api/events', params={'code': 'abc123'}).json()['events'][0]['level'] == 2
    assert client.get('/api/vapid').json() == {'public_key': 'pub'}


def test_gone_subscription_is_dropped_and_laptop_can_restore(service, monkeypatch):
    module, client, _ = service
    client.post('/api/subscribe', json={'code': 'abc123', 'subscription': SUB})
    assert client.get('/api/subscriptions', headers=AUTH).json() == {'subscriptions': [SUB]}

    class Gone:
        status_code = 410

    def expired(**kw):
        raise module.WebPushException('gone', response=Gone())
    monkeypatch.setattr(module, 'webpush', expired)
    out = client.post('/api/alert', json={'level': 2, 'title': 't', 'body': 'b'}, headers=AUTH).json()
    assert out == {'sent': 0, 'subscribers': 0}
    assert client.post('/api/restore', json={'subscriptions': [SUB, {'endpoint': 'bad'}]}).status_code == 401
    assert client.post('/api/restore', json={'subscriptions': [SUB, {'endpoint': 'bad'}]}, headers=AUTH).json() == {'subscribers': 1}


def test_page_and_service_worker_are_served(service):
    _, client, _ = service
    assert 'Bật thông báo' in client.get('/').text
    assert "addEventListener('push'" in client.get('/sw.js').text
    assert client.get('/api/health').json() == {'ok': True, 'subscribers': 0}
