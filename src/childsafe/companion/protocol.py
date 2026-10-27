"""Versioned, image-free messages shared by Python and the Android companion.

AES-GCM keys are generated on the monitor and shared out of band in the pairing
QR. They are never sent to the relay. Transport must additionally use HTTPS.
"""
from __future__ import annotations

import base64
import json
import secrets
from typing import Literal

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, field_validator


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def unb64(value: str) -> bytes:
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


class PersonState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    person_id: int = Field(ge=0)
    risk: Literal["normal", "caution", "danger"]
    observation: Literal["observable", "degraded", "unobservable"]
    reason: str = Field(max_length=1000)


class CameraState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=1, max_length=80)
    kind: Literal["status", "alert"]
    captured_at_ms: int = Field(ge=0)
    risk: Literal["normal", "caution", "danger"]
    observation: Literal["observable", "degraded", "unobservable"]
    reason: str = Field(max_length=1000)
    people: list[PersonState] = Field(default_factory=list, max_length=4)


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    epoch: str = Field(min_length=1, max_length=80)
    sequence: int = Field(ge=0, le=2**53 - 1)
    expires_at_ms: int = Field(ge=0)
    nonce: str = Field(min_length=16, max_length=16)
    ciphertext: str = Field(min_length=22, max_length=22000)

    @field_validator("nonce")
    @classmethod
    def valid_nonce(cls, value: str) -> str:
        if len(unb64(value)) != 12:
            raise ValueError("AES-GCM nonce must be 12 bytes")
        return value

    @field_validator("ciphertext")
    @classmethod
    def valid_ciphertext(cls, value: str) -> str:
        if not 16 <= len(unb64(value)) <= 16384:
            raise ValueError("Invalid ciphertext size")
        return value

    def aad(self, room_id: str, camera_id: str) -> bytes:
        # This exact UTF-8 string is also implemented by the Android client.
        return (f"childsafe/1/{room_id}/{camera_id}/{self.epoch}/"
                f"{self.sequence}/{self.expires_at_ms}").encode("utf-8")


def seal(key: bytes, state: CameraState, *, room_id: str, camera_id: str,
         epoch: str, sequence: int, expires_at_ms: int) -> Envelope:
    if len(key) != 32:
        raise ValueError("Pairing key must contain 32 bytes")
    nonce = secrets.token_bytes(12)
    # Construct AAD without temporarily constructing an invalid Envelope.
    aad = (f"childsafe/1/{room_id}/{camera_id}/{epoch}/"
           f"{sequence}/{expires_at_ms}").encode("utf-8")
    plaintext = state.model_dump_json().encode("utf-8")
    return Envelope(epoch=epoch, sequence=sequence, expires_at_ms=expires_at_ms,
                    nonce=b64(nonce), ciphertext=b64(AESGCM(key).encrypt(nonce, plaintext, aad)))


def open_envelope(key: bytes, envelope: Envelope, *, room_id: str,
                  camera_id: str) -> CameraState:
    if len(key) != 32:
        raise ValueError("Pairing key must contain 32 bytes")
    plaintext = AESGCM(key).decrypt(unb64(envelope.nonce), unb64(envelope.ciphertext),
                                     envelope.aad(room_id, camera_id))
    return CameraState.model_validate(json.loads(plaintext))
