"""HTTP client for streaming captured gameplay to the federation hub.

Bounded asyncio.Queue + background httpx worker so capture never
stalls the gameplay loop. Hot path is `put_nowait`; slow HTTP I/O is
out-of-band. Hub down → queue caps at 1000, drops oldest.

Vendored RELAY-binary encoder must stay byte-identical to
`federated.protocol.encode_arrays`. Roundtrip-tested in
test_federation_client.py.

Env: AURA_FEDERATION_URL + AURA_INGEST_TOKEN. Either unset → ingest
disabled (`from_env` returns None).
"""
from __future__ import annotations

import asyncio
import logging
import os
import struct
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --- RELAY binary encoder (vendored from federated.protocol) ---

_MAGIC = b"RELAY\x00\x00\x00"
_DTYPE_FLOAT32 = 0
_DTYPE_INT8 = 1
_DTYPE_FLOAT16 = 2


def encode_arrays(arrays: Dict[str, np.ndarray]) -> bytes:
    """Pack a dict of float32 numpy arrays into the RELAY binary
    format. Mirrors `federated.protocol.encode_arrays` (float32-only;
    the data tap never produces fp16 / int8)."""
    chunks = [_MAGIC, struct.pack("<I", len(arrays))]
    for name, arr in arrays.items():
        if arr.dtype != np.dtype("float32"):
            raise TypeError(
                f"federation_client: only float32 supported, "
                f"got {arr.dtype} on {name!r}")
        arr_le = arr.astype("<f4", copy=False)
        name_bytes = name.encode("utf-8")
        chunks.append(struct.pack("<I", len(name_bytes)))
        chunks.append(name_bytes)
        chunks.append(struct.pack("<BB", _DTYPE_FLOAT32, arr.ndim))
        if arr.ndim > 0:
            chunks.append(struct.pack(f"<{arr.ndim}I", *arr.shape))
        data_bytes = arr_le.tobytes()
        chunks.append(struct.pack("<I", len(data_bytes)))
        chunks.append(data_bytes)
    return b"".join(chunks)


# --- Async ingest client ---

class IngestClient:
    """Lifecycle: `start()` after the loop is running, `enqueue()` from
    anywhere, `stop()` at shutdown. Worker retries with exponential
    backoff on hub failures; `enqueue` is never blocked."""

    def __init__(
        self,
        *,
        base_url: str,
        game_id: str,
        token: str,
        max_queue_size: int = 1000,
        request_timeout: float = 10.0,
        retry_initial_seconds: float = 1.0,
        retry_max_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.game_id = game_id
        self.token = token
        self.request_timeout = request_timeout
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds

        self._queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=max_queue_size)
        self._task: Optional[asyncio.Task] = None
        self._stop_signal: Optional[asyncio.Event] = None

        self.n_enqueued = 0
        self.n_succeeded = 0
        self.n_failed = 0
        self.n_dropped = 0

    @property
    def ingest_url(self) -> str:
        return f"{self.base_url}/games/{self.game_id}/ingest"

    @classmethod
    def from_env(cls, game_id: str, **overrides) -> Optional["IngestClient"]:
        """Returns None when AURA_FEDERATION_URL or AURA_INGEST_TOKEN
        is unset — callers treat None as 'ingest disabled'."""
        url = os.environ.get("AURA_FEDERATION_URL", "").strip()
        token = os.environ.get("AURA_INGEST_TOKEN", "").strip()
        if not url or not token:
            return None
        return cls(base_url=url, game_id=game_id, token=token, **overrides)

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop_signal = asyncio.Event()
        self._task = asyncio.create_task(
            self._worker(), name=f"ingest-{self.game_id}")

    async def stop(self, drain_timeout: float = 5.0) -> None:
        """Drain remaining queue, then exit. Bounded drain prevents
        shutdown hangs if the hub is unreachable."""
        if self._stop_signal is None or self._task is None:
            return
        self._stop_signal.set()
        try:
            await asyncio.wait_for(self._task, timeout=drain_timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
            logger.warning(
                "ingest: worker drain timed out at %.1fs; %d items "
                "dropped on shutdown",
                drain_timeout, self._queue.qsize())

    def enqueue(self, emb: np.ndarray, actions: np.ndarray) -> bool:
        """Non-blocking. Drops oldest on overflow. Returns False on encode
        failure or drop, True when enqueued."""
        try:
            blob = encode_arrays({"emb": emb, "actions": actions})
        except Exception:
            logger.exception("ingest: encode failed (skipping sample)")
            return False
        self.n_enqueued += 1
        try:
            self._queue.put_nowait(blob)
            return True
        except asyncio.QueueFull:
            try:
                _ = self._queue.get_nowait()
                self.n_dropped += 1
                self._queue.put_nowait(blob)
                return True
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self.n_dropped += 1
                return False

    async def _post_blob(self, blob: bytes) -> bool:
        # Lazy import so callers without httpx installed can still
        # import this module — they'll just get None from from_env().
        try:
            import httpx
        except ImportError:
            logger.error("ingest: httpx not installed; ingest disabled")
            return False
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as cl:
                resp = await cl.post(
                    self.ingest_url, content=blob,
                    headers={"Authorization": f"Bearer {self.token}",
                             "Content-Type": "application/octet-stream"},
                )
            if resp.status_code // 100 == 2:
                return True
            logger.warning(
                "ingest: hub returned %d: %s",
                resp.status_code, resp.text[:200])
            return False
        except Exception as e:
            logger.warning("ingest: POST failed: %s", e)
            return False

    async def _worker(self) -> None:
        backoff = self.retry_initial_seconds
        while True:
            stopping = self._stop_signal is not None and self._stop_signal.is_set()
            if stopping and self._queue.empty():
                return
            try:
                blob = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            ok = await self._post_blob(blob)
            if ok:
                self.n_succeeded += 1
                backoff = self.retry_initial_seconds
            else:
                self.n_failed += 1
                if stopping:
                    continue
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.retry_max_seconds)
