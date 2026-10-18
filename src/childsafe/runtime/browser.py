"""One local browser camera lease with conservative clock bounds, not sensor latency."""
import math
import secrets
import time
from contextlib import contextmanager
from threading import RLock


class FrameRejected(ValueError):
    pass


class SessionEpoch:
    """Serialize cancellation with publication, without blocking stop on inference."""
    def __init__(self, verify):
        self.lock, self.cancelled, self.verify = RLock(), False, verify

    @contextmanager
    def publication(self):
        with self.lock:
            if self.cancelled:
                raise FrameRejected('camera session ended during processing')
            self.verify()
            yield

    def cancel(self):
        with self.lock:
            self.cancelled = True


class BrowserLease:
    def __init__(self, stale_s=.25, lease_s=30., clock=time.perf_counter):
        self.stale_s, self.lease_s, self.clock = stale_s, lease_s, clock
        self.token = None
        self.synced = False

    def start(self, client_send):
        now = self.clock()
        if not math.isfinite(client_send):
            raise FrameRejected('invalid clock')
        if self.token and now < self.expires:
            raise FrameRejected('another camera session is active')
        self.token = secrets.token_urlsafe(24)
        self.sent, self.client_send, self.expires = now, client_send, now+self.lease_s
        self.sequence, self.capture, self.synced = -1, -math.inf, False
        self.reset_floor = -math.inf
        return {'token':self.token, 'server_t':now}

    def sync(self, token, client_receive):
        self.check(token)
        if not math.isfinite(client_receive) or client_receive < self.client_send:
            raise FrameRejected('invalid clock interval')
        self.uncertainty = (client_receive-self.client_send)/2
        if self.uncertainty > self.stale_s/4:
            raise FrameRejected('clock uncertainty too large; retry session')
        self.offset = self.sent-(client_receive+self.client_send)/2
        self.synced = True
        return {'clock_uncertainty_s':self.uncertainty, 'lease_s':self.lease_s}

    def check(self, token):
        if not token or token != self.token or self.clock() >= self.expires:
            raise FrameRejected('camera session expired or not owned')

    def renew(self, token, client_send):
        self.check(token)
        if not math.isfinite(client_send):
            raise FrameRejected('invalid clock')
        self.sent, self.client_send = self.clock(), client_send
        self.expires = self.sent+self.lease_s
        self.synced = False
        return {'token':self.token, 'server_t':self.sent}

    def frame(self, token, sequence, capture):
        self.check(token)
        if not self.synced or not isinstance(sequence,int) or sequence <= self.sequence:
            raise FrameRejected('unverified clock or duplicate frame')
        if not math.isfinite(capture) or capture <= self.capture:
            raise FrameRejected('duplicate or invalid capture timestamp')
        mapped = capture+self.offset
        if mapped-self.uncertainty < self.reset_floor:
            raise FrameRejected('frame predates reset')
        age = self.clock()-mapped
        if age < -self.uncertainty-.001 or age+self.uncertainty > self.stale_s:
            raise FrameRejected('expired or future frame')
        self.sequence, self.capture = sequence, capture
        return mapped+self.stale_s-self.uncertainty, max(0.,age+self.uncertainty)

    def end(self, token):
        self.check(token)
        self.token, self.synced = None, False
