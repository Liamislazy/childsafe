from childsafe.companion.forwarder import CameraForwarder


class Publisher:
    def __init__(self): self.messages = []
    def submit(self, state, **kwargs): self.messages.append((state, kwargs))


def result(level=0, observation="observable", notify=False):
    return dict(level=level, observation=observation, notify=notify, reason="test", people=[])


def test_alerts_immediate_status_throttled_and_no_image_payload():
    now=[10.0]; pub=Publisher(); f=CameraForwarder(pub,clock=lambda:now[0])
    f.result(result(),10)
    now[0]=10.1;f.result(result(),10.1)
    assert len(pub.messages)==1
    now[0]=10.2;f.result(result(1),10.2)
    assert pub.messages[-1][0].kind=="alert"
    now[0]=10.3;f.result(result(2),10.3)
    assert pub.messages[-1][0].risk=="danger"
    assert "people" in pub.messages[-1][0].model_dump()
    assert "image" not in pub.messages[-1][0].model_dump()


def test_stale_high_risk_result_cannot_create_new_event():
    pub=Publisher();f=CameraForwarder(pub,clock=lambda:10.0)
    assert not f.result(result(2,notify=True),9.0)
    state=pub.messages[-1][0]
    assert state.kind=="status" and state.risk=="normal" and state.observation=="unobservable"


def test_loss_expires_evidence_and_reset_does_not_transfer_history():
    now=[10.0];pub=Publisher();f=CameraForwarder(pub,clock=lambda:now[0],status_interval=0)
    f.result(result(2),10)
    now[0]=10.5;f.unavailable("mouth lost")
    assert pub.messages[-1][0].risk=="danger"
    now[0]=10.9;f.unavailable("mouth lost")
    assert pub.messages[-1][0].risk=="normal" and pub.messages[-1][0].observation=="unobservable"
    f.reset()
    now[0]=11.0;f.result(result(2),11)
    assert pub.messages[-1][0].kind=="alert"
