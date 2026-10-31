import base64
import concurrent.futures
import json

import pytest
from cryptography.exceptions import InvalidTag
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from childsafe.companion.protocol import CameraState, Envelope, open_envelope, seal
from childsafe.companion.relay import RelayStore, create_relay


KEY = bytes(range(32))


def state(**kwargs):
    return CameraState(event_id="event-1", kind="alert", captured_at_ms=1000000,
                       risk="danger", observation="degraded", reason="visible object near mouth", **kwargs)


@pytest.fixture
def setup(tmp_path):
    now = [1000.0]
    store = RelayStore(tmp_path / "relay.sqlite3", clock=lambda: now[0], history_limit=3)
    room = store.create_room()
    invite = store.invite(room["room_id"], room["monitor_token"])
    camera = store.join(invite["invitation"], "Test camera")
    session = store.start(camera["camera_id"], camera["camera_token"])
    yield store, now, room, invite, camera, session
    store.close()


def envelope(setup, seq=0, **kwargs):
    store, _, room, _, camera, session = setup
    return seal(KEY, state(), room_id=room["room_id"], camera_id=camera["camera_id"],
                epoch=session["epoch"], sequence=seq, expires_at_ms=store.now() + 3000, **kwargs)


def publish(setup, seq=0):
    store, _, _, _, camera, _ = setup
    return store.publish(camera["camera_id"], camera["camera_token"], envelope(setup, seq))


def test_crypto_roundtrip_and_no_cleartext(setup):
    _, _, room, _, camera, _ = setup
    message = envelope(setup)
    assert open_envelope(KEY, message, room_id=room["room_id"], camera_id=camera["camera_id"]) == state()
    assert "visible object" not in message.model_dump_json()
    assert "danger" not in message.model_dump_json()
    assert message.nonce != envelope(setup).nonce


@pytest.mark.parametrize("field,value", [("sequence", 4), ("expires_at_ms", 5000), ("epoch", "old")])
def test_authenticated_metadata_cannot_be_changed(setup, field, value):
    _, _, room, _, camera, _ = setup
    message = envelope(setup).model_copy(update={field: value})
    with pytest.raises(InvalidTag):
        open_envelope(KEY, message, room_id=room["room_id"], camera_id=camera["camera_id"])


def test_wrong_key_wrong_camera_rejected(setup):
    _, _, room, _, camera, _ = setup
    message = envelope(setup)
    with pytest.raises(InvalidTag):
        open_envelope(b"x" * 32, message, room_id=room["room_id"], camera_id=camera["camera_id"])
    with pytest.raises(InvalidTag):
        open_envelope(KEY, message, room_id=room["room_id"], camera_id="other")


def test_no_images_or_unknown_quality_in_payload():
    with pytest.raises(ValidationError):
        state(video="base64")
    with pytest.raises(ValidationError):
        CameraState.model_validate({**state().model_dump(), "observation": "safe"})


def test_invitation_one_use_and_expires(setup):
    store, now, room, invite, _, _ = setup
    with pytest.raises(HTTPException):
        store.join(invite["invitation"], "Attacker")
    invite = store.invite(room["room_id"], room["monitor_token"])
    now[0] += 121
    with pytest.raises(HTTPException):
        store.join(invite["invitation"], "Too late")


def test_concurrent_pairing_consumes_once(setup):
    store, _, room, _, _, _ = setup
    invite = store.invite(room["room_id"], room["monitor_token"])
    def join(_):
        try:
            return store.join(invite["invitation"], "One")
        except HTTPException:
            return None
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        assert sum(r is not None for r in pool.map(join, range(2))) == 1


def test_separate_roles_and_households(setup):
    store, _, room, _, camera, _ = setup
    other = store.create_room()
    with pytest.raises(HTTPException):
        store.read(room["room_id"], camera["camera_token"], 0)
    with pytest.raises(HTTPException):
        store.read(room["room_id"], other["monitor_token"], 0)
    with pytest.raises(HTTPException):
        store.publish(camera["camera_id"], room["monitor_token"], envelope(setup))


def test_sequence_session_and_expiry_guards(setup):
    store, now, _, _, camera, _ = setup
    publish(setup)
    for _ in range(2):
        with pytest.raises(HTTPException):
            publish(setup)
    old = envelope(setup, 1)
    now[0] += 4
    with pytest.raises(HTTPException):
        store.publish(camera["camera_id"], camera["camera_token"], old)
    store.start(camera["camera_id"], camera["camera_token"])
    with pytest.raises(HTTPException):
        publish(setup, 2)


def test_offline_and_reconnect_do_not_reuse_old_state(setup):
    store, now, room, _, camera, _ = setup
    read = lambda: store.read(room["room_id"], room["monitor_token"], 0, snapshot=True)
    assert not read()["cameras"][0]["connected"]
    publish(setup)
    assert read()["cameras"][0]["connected"]
    now[0] += 16
    assert not read()["cameras"][0]["connected"]
    old_epoch = read()["cameras"][0]["epoch"]
    store.start(camera["camera_id"], camera["camera_token"])
    snapshot = read()
    assert not snapshot["cameras"][0]["connected"]
    assert snapshot["cameras"][0]["epoch"] != old_epoch


def test_bounded_history_gap_and_snapshot(setup):
    store, _, room, _, _, _ = setup
    first = publish(setup)["cursor"]
    for seq in range(1, 6):
        publish(setup, seq)
    result = store.read(room["room_id"], room["monitor_token"], first)
    assert result["history_gap"] and len(result["messages"]) == 3
    snapshot = store.read(room["room_id"], room["monitor_token"], 0, snapshot=True)
    assert len(snapshot["messages"]) == 1
    assert snapshot["messages"][0]["envelope"]["sequence"] == 5


def test_other_rooms_do_not_cause_false_history_gap(setup):
    store, _, room, _, _, _ = setup
    first = publish(setup)["cursor"]
    other = store.create_room()
    inv = store.invite(other["room_id"], other["monitor_token"])
    cam = store.join(inv["invitation"], "Other")
    epoch = store.start(cam["camera_id"], cam["camera_token"])["epoch"]
    for seq in range(10):
        e = seal(KEY, state(), room_id=other["room_id"], camera_id=cam["camera_id"],
                 epoch=epoch, sequence=seq, expires_at_ms=store.now() + 3000)
        store.publish(cam["camera_id"], cam["camera_token"], e)
    publish(setup, 1)
    assert not store.read(room["room_id"], room["monitor_token"], first)["history_gap"]


def test_revocation_removes_access_and_history(setup):
    store, _, room, _, camera, _ = setup
    publish(setup)
    store.revoke(room["room_id"], room["monitor_token"], camera["camera_id"])
    with pytest.raises(HTTPException):
        publish(setup, 1)
    snapshot = store.read(room["room_id"], room["monitor_token"], 0, snapshot=True)
    assert snapshot["cameras"] == snapshot["messages"] == []


def test_database_does_not_contain_tokens_or_plaintext(setup):
    store, _, room, invite, camera, _ = setup
    publish(setup)
    dump = "\n".join(store.db.iterdump())
    for value in [room["monitor_token"], invite["invitation"], camera["camera_token"], "visible object"]:
        assert value not in dump


def test_http_bounds_validation_and_unauthorized(tmp_path):
    with TestClient(create_relay(RelayStore(tmp_path / "http.sqlite3"))) as client:
        assert client.post("/v1/rooms", content=b"x" * 32769).status_code == 413
        room = client.post("/v1/rooms").json()
        endpoint = f'/v1/rooms/{room["room_id"]}/events'
        assert client.get(endpoint).status_code == 401
        headers = {"Authorization": f'Bearer {room["monitor_token"]}'}
        assert client.get(endpoint, headers=headers).status_code == 200
        assert client.get(endpoint, headers=headers).headers["cache-control"] == "no-store"
        assert client.get(endpoint + "?after=-1", headers=headers).status_code == 422
        assert client.get(endpoint + "?after=10000", headers=headers).status_code == 422
        assert client.post("/v1/join", json={"invitation": "x", "name": "x"}).status_code == 422


def test_pairing_rate_limited(tmp_path):
    with TestClient(create_relay(RelayStore(tmp_path / "http.sqlite3"))) as client:
        for _ in range(10):
            assert client.post("/v1/rooms").status_code == 201
        assert client.post("/v1/rooms").status_code == 429
