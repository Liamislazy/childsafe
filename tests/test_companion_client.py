import json
from pathlib import Path

import httpx
import pytest

from childsafe.companion.client import EventPublisher, read_pairing, state_from_result, tls_context
from childsafe.companion.protocol import CameraState, Envelope, b64, open_envelope, unb64
from childsafe.companion.relay import RelayStore


KEY = bytes(range(32))  # Public test vector only; never used by the application.


def state(kind="status", event="test"):
    return CameraState(event_id=event, kind=kind, captured_at_ms=1000000,
                       risk="danger", observation="degraded", reason="test")


@pytest.fixture
def publisher(tmp_path):
    now = [1000.0]
    store = RelayStore(tmp_path / "client.sqlite3", clock=lambda: now[0])
    room = store.create_room()
    invite = store.invite(room["room_id"], room["monitor_token"])
    camera = store.join(invite["invitation"], "Webcam test")
    calls = []
    def handle(request):
        calls.append(request)
        token = request.headers["authorization"].removeprefix("Bearer ")
        if request.url.path.endswith("/session"):
            result = store.start(camera["camera_id"], token)
        else:
            result = store.publish(camera["camera_id"], token,
                                    Envelope.model_validate(json.loads(request.content)))
        return httpx.Response(201, json=result)
    client = httpx.Client(base_url="https://test/", transport=httpx.MockTransport(handle))
    cfg = {"room_id":room["room_id"],"camera_id":camera["camera_id"],"token":camera["camera_token"],
           "key":b64(KEY),"relay_url":"https://test"}
    pub = EventPublisher(cfg, clock=lambda: now[0], client=client)
    yield pub, store, room, now, calls
    pub.close()
    store.close()


def test_publisher_to_relay_to_monitor_decrypt(publisher):
    pub, store, room, now, calls = publisher
    pub._send(state("alert"), now[0] + 2)
    result = store.read(room["room_id"], room["monitor_token"], 0)
    message = result["messages"][0]
    decoded = open_envelope(KEY, Envelope.model_validate(message["envelope"]),
                            room_id=room["room_id"],camera_id=pub.config["camera_id"])
    assert decoded == state("alert")
    assert len(calls) == 2
    assert all(pub.config["token"] not in str(c.url) for c in calls)


def test_expired_states_not_sent_after_handshake(publisher):
    pub, _, _, now, calls = publisher
    pub._send(state(), now[0]-1)
    assert len(calls) == 1  # Handshake only, never an expired event.
    assert pub.dropped == 1


def test_stop_during_handshake_prevents_event_publish(publisher):
    pub, _, _, now, calls = publisher
    pub.stopped = True
    pub._send(state(), now[0]+2)
    assert len(calls) == 1


def test_pending_queue_is_bounded_prioritizes_alerts_and_coalesces_status(publisher):
    pub, _, _, now, _ = publisher
    for i in range(20):
        pub.submit(state("status",str(i)),deadline=now[0]+2)
    for i in range(10):
        pub.submit(state("alert",str(i)),deadline=now[0]+2)
    assert len(pub.alerts)==4 and pub.latest[0].event_id=="19"
    assert [pub._take()[0].event_id for _ in range(4)] == ["6","7","8","9"]
    assert pub._take()[0].event_id=="19"


def test_invalid_and_expired_submit_does_not_fill_queue(publisher):
    pub, _, _, now, _ = publisher
    assert not pub.submit(state(),deadline=now[0])
    assert not pub.alerts and pub.latest is None


def test_result_adapter_never_sends_image_coordinates():
    result = {"level":2,"observation":"degraded","notify":True,"reason":"test",
              "objects":[{"mask":[[1,2]]}],"image":"never export", "people":[
                  {"track_id":5,"level":2,"observation":"unobservable","reason":"mouth lost"}]}
    payload=state_from_result(result,captured_at_ms=1000,event_id="id")
    assert payload.people[0].person_id==5 and payload.kind=="alert"
    assert payload.people[0].observation=="unobservable"
    assert "mask" not in payload.model_dump_json() and "never export" not in payload.model_dump_json()


@pytest.mark.parametrize("url",["http://192.168.1.10:8443","https://user:pass@example.com","https://example.com/?token=x","https://example.com/#key"])
def test_no_cleartext_or_url_credentials(url):
    with pytest.raises(ValueError): tls_context(url)


def test_pairing_file_size_and_schema(tmp_path):
    path=tmp_path/"pair.txt"
    path.write_text("childsafe-pair:"+b64(json.dumps({"version":1,"key":b64(KEY)}).encode()))
    assert read_pairing(path)["version"]==1
    path.write_text("x"*8193)
    with pytest.raises(ValueError): read_pairing(path)


def test_shared_python_android_crypto_fixture():
    path=Path(__file__).parent/"fixtures"/"companion-v1.properties"
    props=dict(line.split("=",1) for line in path.read_text().splitlines() if line and not line.startswith("#"))
    envelope=Envelope(epoch=props["epoch"],sequence=int(props["sequence"]),expires_at_ms=int(props["expires_at_ms"]),
                      nonce=props["nonce"],ciphertext=props["ciphertext"])
    actual=open_envelope(unb64(props["test_key"]),envelope,room_id=props["room_id"],camera_id=props["camera_id"])
    assert actual.model_dump_json().encode()==unb64(props["plaintext"])
