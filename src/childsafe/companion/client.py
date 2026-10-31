"""HTTPS camera publisher with bounded pending events and no video payloads."""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit
import uuid

import httpx

from .protocol import CameraState, PersonState, seal, unb64


def tls_context(base_url: str, fingerprint: str = "") -> ssl.SSLContext:
    parts = urlsplit(base_url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("HTTPS relay URL without credentials/query/fragment required")
    if not fingerprint:
        return ssl.create_default_context()
    if len(fingerprint) != 64 or any(c not in "0123456789abcdefABCDEF" for c in fingerprint):
        raise ValueError("Certificate fingerprint must be 64 hex characters")
    # Probe sends no HTTP request or credentials. Trust the presented certificate
    # only after checking its exact fingerprint from the trusted pairing QR.
    probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE
    with socket.create_connection((parts.hostname, parts.port or 443), timeout=5) as raw:
        with probe.wrap_socket(raw, server_hostname=parts.hostname) as connection:
            cert = connection.getpeercert(binary_form=True)
    if hashlib.sha256(cert).hexdigest() != fingerprint.lower():
        raise ValueError("Relay certificate fingerprint mismatch")
    # Subsequent authenticated requests still verify hostname, validity and the
    # pinned certificate. A trust-all SSL context is never used for HTTP traffic.
    return ssl.create_default_context(cadata=ssl.DER_cert_to_PEM_cert(cert))


def read_pairing(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    if len(raw) > 8192:
        raise ValueError("Pairing data exceeds size limit")
    text = raw.decode("utf-8-sig").strip()
    data = json.loads(unb64(text.removeprefix("childsafe-pair:")).decode()) if text.startswith("childsafe-pair:") else json.loads(text)
    if data.get("version") != 1 or len(unb64(data["key"])) != 32:
        raise ValueError("Unsupported pairing data")
    return data


def state_from_result(result: dict, *, captured_at_ms: int, event_id: str | None = None,
                      alert: bool | None = None) -> CameraState:
    names = ["normal", "caution", "danger"]
    level = int(result["level"])
    if level not in range(3):
        raise ValueError("Invalid risk level")
    people = [PersonState(person_id=int(p.get("track_id", p.get("id"))), risk=names[int(p["level"])],
                          observation=p["observation"], reason=str(p.get("reason", ""))[:1000])
              for p in result.get("people", [])]
    return CameraState(event_id=event_id or uuid.uuid4().hex,
                       kind="alert" if (result.get("notify", False) if alert is None else alert) else "status",
                       captured_at_ms=captured_at_ms, risk=names[level], observation=result["observation"],
                       reason=str(result.get("reason", ""))[:1000], people=people)


class EventPublisher:
    """Nonblocking inference-side submit. Expired work is discarded, never replayed.

    Max four pending alerts plus one latest status. A stalled network cannot
    accumulate camera frames or repeatedly republish a cached normal result.
    """
    def __init__(self, config: dict, *, clock=time.monotonic, client=None):
        self.config, self.clock = config, clock
        self.key = unb64(config["key"])
        if len(self.key) != 32:
            raise ValueError("Invalid encryption key")
        self.client = client or httpx.Client(base_url=config["relay_url"].rstrip("/") + "/",
            verify=tls_context(config["relay_url"], config.get("certificate_pin", "")),
            timeout=5, trust_env=False, follow_redirects=False)
        self.condition = threading.Condition()
        self.alerts = deque(maxlen=4)
        self.latest = None
        self.stopped = False
        self.last_error = ""
        self.dropped = 0
        self.epoch = None
        self.sequence = 0
        self.remote_ms = self.local_t = 0
        self.thread = threading.Thread(target=self._run, name="childsafe-event-publisher", daemon=True)

    @classmethod
    def pair(cls, pairing: dict, *, client=None):
        owned = client is None
        client = client or httpx.Client(base_url=pairing["relay_url"].rstrip("/") + "/",
            verify=tls_context(pairing["relay_url"], pairing.get("certificate_pin", "")), timeout=5, trust_env=False)
        try:
            response = client.post("v1/join", json={"invitation": pairing["invitation"], "name": "Webcam"})
            response.raise_for_status()
            joined = response.json()
            if joined["room_id"] != pairing["room_id"]:
                raise ValueError("Paired room mismatch")
            return {"version": 1, "relay_url": pairing["relay_url"], "room_id": joined["room_id"],
                    "camera_id": joined["camera_id"], "token": joined["camera_token"], "key": pairing["key"],
                    "certificate_pin": pairing.get("certificate_pin", "")}
        finally:
            if owned:
                client.close()

    def start(self):
        self.thread.start()
        return self

    def submit(self, state: CameraState, *, deadline: float):
        with self.condition:
            if self.stopped or self.clock() >= deadline:
                self.dropped += 1
                return False
            item = (state, deadline)
            if state.kind == "alert":
                if len(self.alerts) == self.alerts.maxlen:
                    self.dropped += 1
                self.alerts.append(item)
            else:
                self.latest = item
            self.condition.notify()
            return True

    def _take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.stopped or self.alerts or self.latest, timeout=.5)
            if self.stopped:
                return None
            if self.alerts:
                return self.alerts.popleft()
            item, self.latest = self.latest, None
            return item

    def _send(self, state, deadline):
        headers = {"Authorization": "Bearer " + self.config["token"]}
        camera = self.config["camera_id"]
        if self.epoch is None:
            sent = self.clock()
            response = self.client.post(f"v1/cameras/{camera}/session", headers=headers)
            response.raise_for_status()
            data = response.json()
            received = self.clock()
            self.epoch, self.sequence = data["epoch"], 0
            # Lower bound for server time. Expiry is shortened by the full
            # handshake uncertainty instead of accepting extra-stale events.
            self.remote_ms, self.local_t = data["server_time_ms"], received
            if received - sent > 2.5:
                raise TimeoutError("Session handshake exceeded delivery budget")
        now = self.clock()
        if self.stopped or now >= deadline:
            self.dropped += 1
            return
        expiry = int(self.remote_ms + (min(deadline, now + 2.5) - self.local_t) * 1000)
        message = seal(self.key, state, room_id=self.config["room_id"], camera_id=camera,
                       epoch=self.epoch, sequence=self.sequence, expires_at_ms=expiry)
        self.sequence += 1
        response = self.client.post(f"v1/cameras/{camera}/events", headers=headers, json=message.model_dump())
        if response.status_code == 409:
            self.epoch = None
        response.raise_for_status()

    def _run(self):
        while not self.stopped:
            item = self._take()
            if item is None:
                continue
            try:
                self._send(*item)
                self.last_error = ""
            except Exception:
                # Never log tokens, pairing QR, private relay URL or response body.
                self.last_error = "Không gửi được sự kiện; kiểm tra kết nối và quyền camera."
                self.epoch = None
                self.dropped += 1
                # This item is discarded. Only newly observed evidence is sent
                # after reconnect; queued items also retain their own deadlines.

    def close(self):
        with self.condition:
            self.stopped = True
            self.alerts.clear()
            self.latest = None
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=11)
        if not self.thread.is_alive():
            self.client.close()
