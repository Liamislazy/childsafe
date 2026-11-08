"""ChildSafe phone alerts, deployed on Render (render.yaml).

The laptop demo posts alert events here: level, reason and time, never images. This service sends
them as Web Push notifications to phones that opened the page and allowed notifications.

State is in memory. Render's free instance sleeps after about 15 minutes without requests and then
forgets the subscriptions, so the laptop pings while the demo runs and restores a saved copy
(src/childsafe/notify.py).

Environment: VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY (scripts/make_vapid_keys.py), ALERT_TOKEN
(the laptop's secret), MONITOR_CODE (typed once on each phone), VAPID_SUBJECT (optional contact
URL; defaults to the service URL).
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pywebpush import WebPushException, webpush

HERE = Path(__file__).parent
NO_CACHE = {'Cache-Control': 'no-store'}
app = FastAPI(title='ChildSafe phone alerts')
lock = threading.Lock()
subscriptions: dict[str, dict] = {}   # push endpoint -> subscription
events: deque = deque(maxlen=20)


class Subscribe(BaseModel):
    code: str
    subscription: dict


class Code(BaseModel):
    code: str


class Alert(BaseModel):
    level: int
    title: str
    body: str


class Restore(BaseModel):
    subscriptions: list[dict]


def _matches(given: str | None, env: str) -> bool:
    secret = os.environ.get(env, '')
    return bool(secret) and given is not None and hmac.compare_digest(given.encode(), secret.encode())


def _laptop(authorization: str | None):
    if not _matches((authorization or '').removeprefix('Bearer '), 'ALERT_TOKEN'):
        raise HTTPException(401, 'bad token')


def _phone(code: str):
    if not _matches(code, 'MONITOR_CODE'):
        raise HTTPException(401, 'Sai mã ghép đôi.')


def _valid(sub: dict) -> bool:
    keys = sub.get('keys') or {}
    return str(sub.get('endpoint', '')).startswith('https://') and bool(keys.get('p256dh')) and bool(keys.get('auth'))


def push(payload: dict, request: Request) -> int:
    """Send to every subscription; drop the ones the push service says are gone."""
    subject = os.environ.get('VAPID_SUBJECT') or str(request.base_url).rstrip('/')
    with lock:
        targets = list(subscriptions.values())
    sent, gone = 0, []
    for sub in targets:
        try:
            webpush(subscription_info=sub, data=json.dumps(payload, ensure_ascii=False),
                    vapid_private_key=os.environ['VAPID_PRIVATE_KEY'], vapid_claims={'sub': subject},
                    ttl=120, headers={'Urgency': 'high'})
            sent += 1
        except WebPushException as error:
            if error.response is not None and error.response.status_code in (404, 410):
                gone.append(sub['endpoint'])
    with lock:
        for endpoint in gone:
            subscriptions.pop(endpoint, None)
    return sent


@app.get('/')
def page():
    return FileResponse(HERE / 'index.html', headers=NO_CACHE)


@app.get('/sw.js')
def service_worker():
    return FileResponse(HERE / 'sw.js', media_type='text/javascript', headers=NO_CACHE)


@app.get('/api/health')
def health():
    with lock:
        return {'ok': True, 'subscribers': len(subscriptions)}


@app.get('/api/vapid')
def vapid():
    return {'public_key': os.environ.get('VAPID_PUBLIC_KEY', '')}


@app.post('/api/subscribe')
def subscribe(body: Subscribe):
    _phone(body.code)
    if not _valid(body.subscription):
        raise HTTPException(422, 'invalid subscription')
    with lock:
        subscriptions[body.subscription['endpoint']] = body.subscription
        return {'subscribers': len(subscriptions)}


@app.post('/api/test')
def test(body: Code, request: Request):
    _phone(body.code)
    return {'sent': push({'level': 0, 'title': 'ChildSafe: thử thông báo',
                          'body': 'Điện thoại này sẽ nhận cảnh báo NGUY HIỂM và CHÚ Ý.'}, request)}


@app.get('/api/events')
def recent(code: str):
    _phone(code)
    with lock:
        return JSONResponse({'events': list(events)[::-1], 'subscribers': len(subscriptions)}, headers=NO_CACHE)


@app.post('/api/alert')
def alert(body: Alert, request: Request, authorization: str | None = Header(None)):
    _laptop(authorization)
    payload = dict(level=body.level, title=body.title[:80], body=body.body[:200], t=time.time())
    with lock:
        events.append(payload)
    sent = push(payload, request)
    with lock:
        return {'sent': sent, 'subscribers': len(subscriptions)}


@app.get('/api/pairing')
def pairing(authorization: str | None = Header(None)):
    """The laptop shows MONITOR_CODE as a QR code, so phones never type it."""
    _laptop(authorization)
    return {'code': os.environ.get('MONITOR_CODE', '')}


@app.get('/api/subscriptions')
def backup(authorization: str | None = Header(None)):
    _laptop(authorization)
    with lock:
        return {'subscriptions': list(subscriptions.values())}


@app.post('/api/restore')
def restore(body: Restore, authorization: str | None = Header(None)):
    _laptop(authorization)
    with lock:
        for sub in body.subscriptions:
            if _valid(sub):
                subscriptions.setdefault(sub['endpoint'], sub)
        return {'subscribers': len(subscriptions)}
