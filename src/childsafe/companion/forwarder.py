"""Map live runtime results to event-only companion updates, with freshness gates."""
import time
import uuid

from .client import state_from_result
from .protocol import CameraState


class CameraForwarder:
    def __init__(self, publisher, *, clock=time.perf_counter, wall_clock=time.time,
                 frame_max_age=.25, status_interval=1., evidence_hold=.8):
        self.publisher, self.clock, self.wall_clock = publisher, clock, wall_clock
        self.frame_max_age, self.status_interval, self.evidence_hold = frame_max_age, status_interval, evidence_hold
        self.last_sent = -float("inf")
        self.last_trusted = -float("inf")
        self.last_level = 0
        self.event_id = uuid.uuid4().hex

    def result(self, result, captured):
        now = self.clock()
        age = now - captured
        if not 0 <= age <= self.frame_max_age:
            self.unavailable("Kết quả ảnh đã cũ; không dùng để tạo bằng chứng mới")
            return False
        level = int(result["level"])
        alert = level > 0 and (level > self.last_level or bool(result.get("notify")))
        if level != self.last_level or alert:
            self.event_id = uuid.uuid4().hex
        self.last_level, self.last_trusted = level, now
        if alert or now - self.last_sent >= self.status_interval:
            state = state_from_result(result, captured_at_ms=int((self.wall_clock()-age)*1000),
                                      event_id=self.event_id, alert=alert)
            self.publisher.submit(state, deadline=captured+2.5)
            self.last_sent = now
        return True

    def unavailable(self, reason):
        now = self.clock()
        if now - self.last_sent < self.status_interval:
            return
        held = now - self.last_trusted <= self.evidence_hold
        if not held:
            self.last_level = 0
        state = CameraState(event_id=self.event_id, kind="status", captured_at_ms=int(self.wall_clock()*1000),
                            risk=["normal", "caution", "danger"][self.last_level],
                            observation="unobservable", reason=reason[:1000])
        self.publisher.submit(state, deadline=now+2.5)
        self.last_sent = now

    def reset(self):
        self.last_trusted = -float("inf")
        self.last_level = 0
        self.event_id = uuid.uuid4().hex
