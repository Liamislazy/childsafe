import time
import numpy as np
from childsafe.alerts.processor import AlertProcessor, DANGER
from childsafe.alerts.temporal import TimeWindow
from childsafe.config import load_config
from childsafe.runtime.source import LatestSlot, FramePacket, VideoSource


def test_slot_retains_only_latest_frame():
    slot = LatestSlot()
    for i in range(100):
        slot.publish(FramePacket(i,float(i),1))
    assert slot.take().frame == 99
    assert slot.take(timeout=0) is None


def test_time_window_never_bridges_long_gap():
    window = TimeWindow()
    assert not window(0,True)
    assert window(.1,True)
    assert not window(1,True)


def test_danger_timing_at_5_10_15_30_fps():
    starts = []
    for fps in [5,10,15,30]:
        cfg = load_config(); cfg['smoothing']['method'] = 'none'
        proc = AlertProcessor(cfg)
        first = None
        for t in np.arange(0,2,1/fps):
            out = proc.step(dict(t=float(t),face_ok=1,hand_ok=1,d_face_width=.1),
                            dict(p_unsafe_raw=.9,p_unsafe_cal=.9,obj_class='knife'))
            if out['level'] == DANGER:
                first = float(t)
                break
        assert first is not None
        starts.append(first)
    for first,fps in zip(starts,[5,10,15,30]):
        assert abs(first-starts[-1]) <= 1/fps+1e-9


def test_reconnect_increments_source_session():
    class Capture:
        def __init__(self): self.i = 0
        def isOpened(self): return True
        def read(self):
            time.sleep(.01)
            self.i += 1
            return (True,np.zeros((2,2,3))) if self.i < 3 else (False,None)
        def release(self): pass
    source = VideoSource(0,reconnect_s=.01,capture_factory=lambda _:Capture()).start()
    sessions = set()
    deadline = time.monotonic()+2
    try:
        while time.monotonic() < deadline and len(sessions) < 2:
            packet = source.read(.1)
            if packet and packet.source_ok:
                sessions.add(packet.session_id)
        assert len(sessions) >= 2
    finally:
        source.close()
