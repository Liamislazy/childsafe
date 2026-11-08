"""Forward alerts to the phone-alert service (services/notify on Render): level, reason, time only.

Never images. Sending runs on its own thread with a small queue, so a slow network never delays
the camera loop. While the demo runs it pings the service every keepalive_s (under Render's
15-minute idle sleep) and keeps a local copy of the phone subscriptions, restoring them if the
service restarted and forgot them.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from pathlib import Path

from childsafe.config import ROOT

STORE = ROOT / 'outputs' / 'phone_alerts' / 'subscriptions.json'   # outputs/ is git-ignored


class PhoneNotifier:
    def __init__(self, url: str, token: str, store: Path = STORE, keepalive_s: float = 600., timeout_s: float = 15.):
        if not url.startswith('https://') and not url.startswith('http://127.0.0.1'):
            raise ValueError('phone alerts need an https:// service URL')
        self.url, self.token, self.store = url.rstrip('/'), token, Path(store)
        self.keepalive_s, self.timeout_s = keepalive_s, timeout_s
        self.queue: queue.Queue = queue.Queue(maxsize=20)
        self.status = dict(sent=0, subscribers=None, last_error='', last_sent_t=None)
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._sender, daemon=True, name='phone-alerts'),
                         threading.Thread(target=self._keepalive, daemon=True, name='phone-alerts-keepalive')]
        for thread in self._threads:
            thread.start()

    def send(self, level: int, title: str, body: str):
        """Queue an alert; when the queue is full the oldest waiting alert is dropped."""
        item = dict(level=int(level), title=title, body=body)
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read() or b'{}')

    def _sender(self):
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                out = self._request('POST', '/api/alert', item)
                if not out.get('subscribers'):
                    self._sync()              # the service may have restarted: restore, then resend
                    out = self._request('POST', '/api/alert', item)
                self.status.update(sent=self.status['sent'] + out.get('sent', 0),
                                   subscribers=out.get('subscribers'), last_error='', last_sent_t=time.time())
            except Exception as error:      # network/DNS/HTTP errors: report, keep the demo running
                self.status['last_error'] = type(error).__name__

    def _sync(self):
        """Back up the service's subscriptions locally, or restore them if it has none."""
        remote = self._request('GET', '/api/subscriptions').get('subscriptions', [])
        if remote:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            self.store.write_text(json.dumps(remote), encoding='utf-8')
        elif self.store.is_file():
            saved = json.loads(self.store.read_text(encoding='utf-8'))
            if saved:
                remote = saved
                self._request('POST', '/api/restore', {'subscriptions': saved})
        self.status['subscribers'] = len(remote)

    def _keepalive(self):
        while not self._stop.is_set():
            try:
                self._sync()
                self.status['last_error'] = ''
            except Exception as error:
                self.status['last_error'] = type(error).__name__
            self._stop.wait(self.keepalive_s)

    def close(self):
        self._stop.set()
