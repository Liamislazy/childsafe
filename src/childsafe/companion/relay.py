"""Small durable relay for encrypted events, not camera video or inference.

Run behind an HTTPS reverse proxy. Credentials go in Authorization headers;
pairing invitations are single-use and short-lived. No cloud service is
provisioned by this module. SQLite holds bounded encrypted history only.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .protocol import Envelope


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class RelayStore:
    def __init__(self, path: str | Path, *, clock=time.time, max_rooms=100,
                 history_limit=512, max_cameras=4):
        self.clock, self.max_rooms = clock, max_rooms
        self.history_limit, self.max_cameras = history_limit, max_cameras
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS rooms (
                id TEXT PRIMARY KEY, owner_hash TEXT UNIQUE NOT NULL,
                discarded_cursor INTEGER NOT NULL DEFAULT 0,
                last_cursor INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY, room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                epoch TEXT, sequence INTEGER NOT NULL DEFAULT -1, last_seen_ms INTEGER);
            CREATE TABLE IF NOT EXISTS invites (
                token_hash TEXT PRIMARY KEY, room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                expires_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                received_ms INTEGER NOT NULL, body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS event_room ON events(room_id,cursor);
        """)

    def now(self):
        return int(self.clock() * 1000)

    def close(self):
        self.db.close()

    def owner(self, room_id, token):
        if not self.db.execute("SELECT 1 FROM rooms WHERE id=? AND owner_hash=?",
                               (room_id, digest(token))).fetchone():
            raise HTTPException(401, "Invalid monitor credentials")

    def camera(self, camera_id, token):
        row = self.db.execute("SELECT * FROM cameras WHERE id=? AND token_hash=?",
                              (camera_id, digest(token))).fetchone()
        if row is None:
            raise HTTPException(401, "Invalid camera credentials")
        return row

    def create_room(self):
        with self.lock, self.db:
            if self.db.execute("SELECT count(*) FROM rooms").fetchone()[0] >= self.max_rooms:
                raise HTTPException(503, "Relay room capacity reached")
            room, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            self.db.execute("INSERT INTO rooms(id,owner_hash) VALUES (?,?)", (room, digest(token)))
            return {"room_id": room, "monitor_token": token, "server_time_ms": self.now()}

    def invite(self, room, token):
        with self.lock, self.db:
            self.owner(room, token)
            invite, expires = secrets.token_urlsafe(32), self.now() + 120_000
            # One active invitation per room; renewing revokes the old QR.
            self.db.execute("DELETE FROM invites WHERE room_id=? OR expires_ms<?", (room, self.now()))
            self.db.execute("INSERT INTO invites VALUES (?,?,?)", (digest(invite), room, expires))
            return {"invitation": invite, "expires_at_ms": expires}

    def join(self, invitation, name):
        with self.lock, self.db:
            row = self.db.execute("SELECT * FROM invites WHERE token_hash=? AND expires_ms>?",
                                  (digest(invitation), self.now())).fetchone()
            if row is None:
                raise HTTPException(401, "Invitation expired or already used")
            room = row["room_id"]
            count = self.db.execute("SELECT count(*) FROM cameras WHERE room_id=?", (room,)).fetchone()[0]
            if count >= self.max_cameras:
                raise HTTPException(409, "Maximum camera count reached")
            camera, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            self.db.execute("DELETE FROM invites WHERE token_hash=?", (digest(invitation),))
            self.db.execute("INSERT INTO cameras(id,room_id,token_hash,name) VALUES (?,?,?,?)",
                            (camera, room, digest(token), name))
            return {"room_id": room, "camera_id": camera, "camera_token": token}

    def start(self, camera, token):
        with self.lock, self.db:
            self.camera(camera, token)
            epoch = secrets.token_hex(16)
            # A reconnect must not revive a previously cached normal state.
            self.db.execute("UPDATE cameras SET epoch=?,sequence=-1,last_seen_ms=NULL WHERE id=?",
                            (epoch, camera))
            return {"epoch": epoch, "server_time_ms": self.now()}

    def publish(self, camera, token, envelope):
        with self.lock, self.db:
            row = self.camera(camera, token)
            now = self.now()
            if envelope.epoch != row["epoch"] or envelope.sequence <= row["sequence"]:
                raise HTTPException(409, "Outdated session or duplicate sequence")
            if not now < envelope.expires_at_ms <= now + 10_000:
                raise HTTPException(409, "Message expired or expiry too far in the future")
            self.db.execute("UPDATE cameras SET sequence=?,last_seen_ms=? WHERE id=?",
                            (envelope.sequence, now, camera))
            result = self.db.execute("INSERT INTO events(room_id,camera_id,received_ms,body) VALUES (?,?,?,?)",
                                    (row["room_id"], camera, now, envelope.model_dump_json()))
            oldest = self.db.execute("""SELECT max(cursor) FROM events WHERE room_id=? AND cursor NOT IN (
                SELECT cursor FROM events WHERE room_id=? ORDER BY cursor DESC LIMIT ?)""",
                (row["room_id"], row["room_id"], self.history_limit)).fetchone()[0]
            self.db.execute("UPDATE rooms SET last_cursor=?,discarded_cursor=max(discarded_cursor,?) WHERE id=?",
                            (result.lastrowid, oldest or 0, row["room_id"]))
            self.db.execute("""DELETE FROM events WHERE room_id=? AND cursor NOT IN (
                SELECT cursor FROM events WHERE room_id=? ORDER BY cursor DESC LIMIT ?)""",
                (row["room_id"], row["room_id"], self.history_limit))
            return {"cursor": result.lastrowid, "server_time_ms": now}

    def read(self, room, token, after, *, snapshot=False):
        with self.lock:
            self.owner(room, token)
            now = self.now()
            cameras = [dict(row) for row in self.db.execute(
                "SELECT id,name,epoch,last_seen_ms FROM cameras WHERE room_id=?", (room,))]
            for camera in cameras:
                seen = camera["last_seen_ms"]
                camera["connected"] = seen is not None and 0 <= now - seen < 15_000
            state = self.db.execute("SELECT discarded_cursor,last_cursor FROM rooms WHERE id=?", (room,)).fetchone()
            if after > state["last_cursor"]:
                raise HTTPException(422, "Cursor is ahead of this room")
            if snapshot:
                rows = self.db.execute("""SELECT * FROM events WHERE room_id=? AND cursor IN (
                    SELECT max(cursor) FROM events WHERE room_id=? GROUP BY camera_id) ORDER BY cursor""",
                    (room, room)).fetchall()
            else:
                rows = self.db.execute("SELECT * FROM events WHERE room_id=? AND cursor>? ORDER BY cursor LIMIT 100",
                                       (room, after)).fetchall()
            gap = bool(after and after < state["discarded_cursor"])
            messages = [{"cursor": row["cursor"], "camera_id": row["camera_id"],
                         "received_at_ms": row["received_ms"], "envelope": json.loads(row["body"])}
                        for row in rows]
            cursor = state["last_cursor"] if snapshot else (rows[-1]["cursor"] if rows else after)
            return {"server_time_ms": now, "cameras": cameras, "messages": messages,
                    "history_gap": gap, "cursor": cursor, "more": cursor < state["last_cursor"]}

    def revoke(self, room, token, camera):
        with self.lock, self.db:
            self.owner(room, token)
            self.db.execute("DELETE FROM cameras WHERE room_id=? AND id=?", (room, camera))

    def delete_room(self, room, token):
        with self.lock, self.db:
            self.owner(room, token)
            self.db.execute("DELETE FROM rooms WHERE id=?", (room,))


class Join(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invitation: str = Field(min_length=32, max_length=80)
    name: str = Field(min_length=1, max_length=60)


def create_relay(store: RelayStore) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_):
        yield
        store.close()

    app = FastAPI(title="ChildSafe encrypted event relay", docs_url=None,
                  redoc_url=None, openapi_url=None, lifespan=lifespan)
    attempts = defaultdict(deque)
    rate_lock = threading.Lock()

    def rate_limit(request):
        # Do not trust X-Forwarded-For from callers. Configure trusted proxies in uvicorn.
        ip = request.client.host if request.client else "local"
        with rate_lock:
            now = time.monotonic()
            for key in list(attempts):
                if not attempts[key] or now - attempts[key][-1] >= 60:
                    del attempts[key]
            q = attempts[ip]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= 10 or len(attempts) > 1000:
                raise HTTPException(429, "Too many pairing attempts")
            q.append(now)

    def bearer(authorization):
        if not authorization or not authorization.startswith("Bearer ") or len(authorization) > 100:
            raise HTTPException(401, "Bearer token required")
        return authorization[7:]

    @app.middleware("http")
    async def bounds(request: Request, call_next):
        # Buffer at most 32 KiB, including chunked requests; never trust Content-Length.
        if request.method in {"POST", "PUT", "PATCH"}:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32768:
                    from starlette.responses import JSONResponse
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/v1/rooms", status_code=201)
    def create(request: Request):
        rate_limit(request)
        return store.create_room()

    @app.post("/v1/rooms/{room}/invitations")
    def invite(room: str, authorization: str | None = Header(None)):
        return store.invite(room, bearer(authorization))

    @app.post("/v1/join", status_code=201)
    def join(body: Join, request: Request):
        rate_limit(request)
        return store.join(body.invitation, body.name)

    @app.post("/v1/cameras/{camera}/session")
    def start(camera: str, authorization: str | None = Header(None)):
        return store.start(camera, bearer(authorization))

    @app.post("/v1/cameras/{camera}/events", status_code=201)
    def publish(camera: str, body: Envelope, authorization: str | None = Header(None)):
        return store.publish(camera, bearer(authorization), body)

    @app.get("/v1/rooms/{room}/events")
    def read(room: str, after: int = 0, authorization: str | None = Header(None)):
        if after < 0:
            raise HTTPException(422, "Cursor must be nonnegative")
        return store.read(room, bearer(authorization), after)

    @app.get("/v1/rooms/{room}/snapshot")
    def snapshot(room: str, authorization: str | None = Header(None)):
        return store.read(room, bearer(authorization), 0, snapshot=True)

    @app.delete("/v1/rooms/{room}/cameras/{camera}", status_code=204)
    def revoke(room: str, camera: str, authorization: str | None = Header(None)):
        store.revoke(room, bearer(authorization), camera)

    @app.delete("/v1/rooms/{room}", status_code=204)
    def delete_room(room: str, authorization: str | None = Header(None)):
        store.delete_room(room, bearer(authorization))

    return app
