"""Real loopback TLS, not a mocked HTTP transport; never opens a camera."""
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
import httpx
import ipaddress
import pytest

from childsafe.companion.client import EventPublisher, tls_context
from childsafe.companion.protocol import CameraState, Envelope, b64, open_envelope


def test_real_tls_pair_publish_decrypt_and_reject_wrong_pin(tmp_path):
    key=ec.generate_private_key(ec.SECP256R1())
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,"ChildSafe TEST ONLY")])
    now=datetime.now(timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1))
          .not_valid_after(now+timedelta(days=1)).add_extension(x509.BasicConstraints(ca=True,path_length=0),True)
          .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),False)
          .sign(key,hashes.SHA256()))
    certificate=tmp_path/"cert.pem";private=tmp_path/"key.pem"
    certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    fingerprint=hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",0));port=probe.getsockname()[1]
    root=Path(__file__).resolve().parents[1]
    process=subprocess.Popen([sys.executable,str(root/"scripts/companion_relay.py"),"--host","127.0.0.1",
        "--port",str(port),"--db",str(tmp_path/"relay.sqlite3"),"--cert",str(certificate),"--key",str(private)],
        cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    url=f"https://127.0.0.1:{port}"
    try:
        deadline=time.monotonic()+10
        while True:
            if process.poll() is not None:
                pytest.fail("Relay exited before TLS test: "+process.stderr.read().decode(errors="replace"))
            try:
                context=tls_context(url,fingerprint)
                break
            except OSError:
                if time.monotonic()>deadline: raise
                time.sleep(.1)
        with pytest.raises(ValueError,match="fingerprint"):
            tls_context(url,"0"*64)
        with httpx.Client(base_url=url+"/",verify=context,trust_env=False,timeout=3) as monitor:
            room=monitor.post("v1/rooms").json()
            auth={"Authorization":"Bearer "+room["monitor_token"]}
            invitation=monitor.post(f'v1/rooms/{room["room_id"]}/invitations',headers=auth).json()
            encryption_key=bytes(range(32)) # Public test vector, never application credentials.
            pairing={"version":1,"relay_url":url,"room_id":room["room_id"],"key":b64(encryption_key),
                     "invitation":invitation["invitation"],"certificate_pin":fingerprint}
            credentials=EventPublisher.pair(pairing)
            publisher=EventPublisher(credentials)
            try:
                state=CameraState(event_id="tls-test",kind="test",captured_at_ms=int(time.time()*1000),
                    risk="normal",observation="unobservable",reason="Manual test, no AI event")
                publisher._send(state,time.monotonic()+2.5)
            finally:
                publisher.close()
            response=monitor.get(f'v1/rooms/{room["room_id"]}/events',headers=auth)
            assert response.status_code==200
            item=response.json()["messages"][0]
            observed=open_envelope(encryption_key,Envelope.model_validate(item["envelope"]),
                room_id=room["room_id"],camera_id=credentials["camera_id"])
            assert observed==state
            assert "Manual test" not in response.text
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stderr.close()
