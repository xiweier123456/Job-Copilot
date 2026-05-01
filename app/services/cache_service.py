from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import redis
except ImportError:  # pragma: no cover - depends on the local environment.
    redis = None


class RawRedisClient:
    """Tiny Redis RESP client used when the optional redis package is unavailable."""

    def __init__(self, url: str, timeout: float = 2.0) -> None:
        parsed = urlparse(url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 6379
        self.password = parsed.password
        self.db = int((parsed.path or "/0").strip("/") or "0")
        self.timeout = timeout

    def _encode_command(self, *parts: Any) -> bytes:
        encoded_parts = [str(part).encode("utf-8") for part in parts]
        payload = [f"*{len(encoded_parts)}\r\n".encode("utf-8")]
        for part in encoded_parts:
            payload.append(f"${len(part)}\r\n".encode("utf-8"))
            payload.append(part + b"\r\n")
        return b"".join(payload)

    def _read_line(self, sock: socket.socket) -> bytes:
        data = bytearray()
        while not data.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                raise ConnectionError("Redis connection closed")
            data.extend(chunk)
        return bytes(data[:-2])

    def _read_response(self, sock: socket.socket) -> Any:
        prefix = sock.recv(1)
        if not prefix:
            raise ConnectionError("Redis connection closed")

        if prefix == b"+":
            return self._read_line(sock).decode("utf-8")
        if prefix == b"-":
            raise RuntimeError(self._read_line(sock).decode("utf-8"))
        if prefix == b":":
            return int(self._read_line(sock))
        if prefix == b"$":
            length = int(self._read_line(sock))
            if length == -1:
                return None
            data = b""
            while len(data) < length:
                data += sock.recv(length - len(data))
            sock.recv(2)
            return data.decode("utf-8")
        if prefix == b"*":
            length = int(self._read_line(sock))
            return [self._read_response(sock) for _ in range(length)]

        raise RuntimeError(f"Unsupported Redis response prefix: {prefix!r}")

    def execute(self, *parts: Any) -> Any:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            if self.password:
                sock.sendall(self._encode_command("AUTH", self.password))
                self._read_response(sock)
            if self.db:
                sock.sendall(self._encode_command("SELECT", self.db))
                self._read_response(sock)
            sock.sendall(self._encode_command(*parts))
            return self._read_response(sock)

    def ping(self) -> bool:
        return self.execute("PING") == "PONG"

    def get(self, key: str) -> str | None:
        return self.execute("GET", key)

    def set(self, key: str, value: str, *, ex: int | None = None, nx: bool = False) -> bool:
        command: list[Any] = ["SET", key, value]
        if ex:
            command.extend(["EX", ex])
        if nx:
            command.append("NX")
        return self.execute(*command) == "OK"

    def delete(self, key: str) -> int:
        return int(self.execute("DEL", key) or 0)


def redis_available() -> bool:
    return bool(settings.redis_enabled and settings.redis_url)


def build_cache_key(*parts: Any) -> str:
    cleaned = [str(part).strip().replace(" ", "_") for part in parts if str(part).strip()]
    return ":".join([settings.redis_key_prefix.strip() or "jobcopilot", *cleaned])


def hash_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


@lru_cache(maxsize=1)
def _sync_client():
    if not redis_available():
        return None
    if redis is not None:
        return redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return RawRedisClient(settings.redis_url)


def _get_sync_client():
    client = _sync_client()
    if client is None:
        return None
    try:
        client.ping()
    except Exception as exc:
        logger.warning("Redis unavailable, falling back to local behavior: %s", exc)
        return None
    return client


async def close_cache() -> None:
    _sync_client.cache_clear()


async def get_json(key: str) -> Any | None:
    raw = await asyncio.to_thread(get_text_sync, key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Redis cached value for %s is not valid JSON", key)
        return None


async def set_json(key: str, value: Any, ttl_seconds: int | None = None) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    await asyncio.to_thread(set_text_sync, key, encoded, ttl_seconds)


def get_text_sync(key: str) -> str | None:
    client = _get_sync_client()
    if client is None:
        return None
    try:
        value = client.get(key)
    except Exception as exc:
        logger.warning("Redis GET failed for %s: %s", key, exc)
        return None
    return str(value) if value is not None else None


def set_text_sync(key: str, value: str, ttl_seconds: int | None = None) -> None:
    client = _get_sync_client()
    if client is None:
        return
    ttl = ttl_seconds or settings.redis_default_ttl_seconds
    try:
        client.set(key, value, ex=ttl)
    except Exception as exc:
        logger.warning("Redis SET failed for %s: %s", key, exc)


def acquire_lock_sync(key: str, value: str, ttl_seconds: int) -> bool | None:
    client = _get_sync_client()
    if client is None:
        return None
    try:
        return bool(client.set(key, value, nx=True, ex=ttl_seconds))
    except Exception as exc:
        logger.warning("Redis lock acquire failed for %s: %s", key, exc)
        return None


def release_lock_sync(key: str, expected_value: str) -> None:
    client = _get_sync_client()
    if client is None:
        return
    try:
        current_value = client.get(key)
        if current_value == expected_value:
            client.delete(key)
    except Exception as exc:
        logger.warning("Redis lock release failed for %s: %s", key, exc)


def delete_sync(key: str) -> None:
    client = _get_sync_client()
    if client is None:
        return
    try:
        client.delete(key)
    except Exception as exc:
        logger.warning("Redis DEL failed for %s: %s", key, exc)
