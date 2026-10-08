"""Latest-frame-only camera/URL source. Timestamps measure internal latency only."""
from dataclasses import dataclass
from threading import Condition, Event, Thread
import time


@dataclass
class FramePacket:
    frame: object
    received_t: float
    session_id: int
    source_ok: bool = True


class LatestSlot:
    def __init__(self):
        self.condition = Condition()
        self.packet = None

    def publish(self, packet):
        with self.condition:
            self.packet = packet
            self.condition.notify_all()

    def take(self, timeout=.2):
        with self.condition:
            if self.packet is None:
                self.condition.wait_for(lambda: self.packet is not None, timeout)
            packet, self.packet = self.packet, None
            return packet


class VideoSource:
    def __init__(self, source, width=1280, height=720, reconnect_s=.5, capture_factory=None):
        self.source, self.width, self.height = source, width, height
        self.reconnect_s = reconnect_s
        self.capture_factory = capture_factory or self._capture
        self.slot, self.stop = LatestSlot(), Event()
        self.session_id, self.cap = 0, None
        self.thread = Thread(target=self._run,daemon=True,name='childsafe-source')

    def _capture(self, source):
        import cv2
        if isinstance(source,int):
            cap = cv2.VideoCapture(source,cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height)
        else:
            cap = cv2.VideoCapture(source,cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,3000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])
        return cap

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.is_set():
            try:
                self.cap = self.capture_factory(self.source)
                if not self.cap.isOpened():
                    raise RuntimeError('source unavailable')
                self.session_id += 1
                while not self.stop.is_set():
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        break
                    self.slot.publish(FramePacket(frame,time.perf_counter(),self.session_id))
            except Exception:
                pass  # Do not disclose camera credentials via backend exception messages.
            finally:
                if self.cap is not None:
                    self.cap.release()
                self.slot.publish(FramePacket(None,time.perf_counter(),self.session_id,False))
            self.stop.wait(self.reconnect_s)

    def read(self, timeout=.2):
        return self.slot.take(timeout)

    def close(self):
        self.stop.set()
        if self.cap is not None:
            self.cap.release()
        if self.thread.is_alive():
            self.thread.join(timeout=4)
