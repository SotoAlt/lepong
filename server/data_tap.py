"""Per-match sliding-window sampler for federation gameplay capture.

Streams `(frame, action)` pairs into the (history+1, history) windows
the v9_relay pool format expects, encodes through the loaded JEPA,
batches B samples per ingest, forwards via an `IngestClient`.

Wire shape (v9_relay manifest):
  emb:     (B, history+1, embed_dim)         float32
  actions: (B, history,   frameskip*2)       float32, action_dim=2

Action slot is the action_dim thrust tiled `frameskip` times,
flattened — same packing `gen_pymunk_v7_episodes.py` uses.

Encoder is called once per pushed frame (not per window) — frame
embeddings are cached so emitting a sliding window is a slice, not
a re-encode. The 4× per-step win matters: on M3 CPU a 4-frame
ViT-Tiny forward is ~150 ms, a 1-frame forward is ~40 ms.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Deque, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Encoder takes a 1-element list of frames and returns a (1, embed_dim)
# numpy array. Returning None signals "skip this frame" (e.g. model
# unloaded mid-match) — the sampler drops the push silently.
FrameEncoder = Callable[[List[np.ndarray]], Optional[np.ndarray]]


def _normalize_emb_to_1d(emb: np.ndarray) -> Optional[np.ndarray]:
    """Coerce an embedding to a 1-D float32 array. Accepts (D,) or
    (1, D); returns None on shape mismatch (caller drops the push)."""
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            logger.warning("data_tap: emb has %d rows, expected 1",
                            arr.shape[0])
            return None
        return arr[0]
    if arr.ndim == 1:
        return arr
    logger.warning("data_tap: emb shape %s — expected (D,) or (1, D)",
                    arr.shape)
    return None


class MatchSampler:
    """Rolling-window sample builder for one match session.

    Usage:
        sampler = MatchSampler(history=3, frameskip=5, action_dim=2,
                                batch_size=2,
                                encoder=encode_frame_via_v9,
                                ingest_client=client)
        sampler.push_initial_frame(frame_0)
        sampler.push_step(thrust, frame_t)   # repeat per env-step
        sampler.flush_partial()              # at match end

    `frame_0` is the post-reset frame, paired with no action.
    `thrust` is the action_dim-length vector held constant over the
    `frameskip` env-ticks between consecutive sampled frames.
    """

    def __init__(
        self,
        *,
        history: int,
        frameskip: int,
        action_dim: int,
        batch_size: int,
        encoder: Optional[FrameEncoder],   # None when callers use push_*_emb
        ingest_client,                     # IngestClient | None
    ) -> None:
        self.history = int(history)
        self.frameskip = int(frameskip)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.encoder = encoder
        self.ingest_client = ingest_client

        # Bounded ring buffers of EMBEDDINGS (not raw frames) + actions.
        # Frames get dropped after encoding — keeps memory tiny over a
        # long match (192 floats vs full RGB image per slot).
        keep = self.history + 1
        self._embs: Deque[np.ndarray] = deque(maxlen=keep)
        self._actions: Deque[np.ndarray] = deque(maxlen=self.history)
        self._batch_buf: List[tuple] = []

        # Track how many windows we've emitted so far. Each new full
        # window emits exactly once; without this counter, every push
        # past the window-fill point would re-emit the same content.
        self._windows_emitted = 0

        self.n_pushes = 0
        self.n_windows = 0
        self.n_batches = 0

    @property
    def window_size(self) -> int:
        return self.history + 1

    def push_initial_frame(self, frame: np.ndarray) -> None:
        self._encode_and_buffer(frame)
        self.n_pushes += 1
        self._maybe_emit_window()

    def push_step(self, thrust: np.ndarray, frame: np.ndarray) -> None:
        slot = self._tile_action(thrust)
        if not self._encode_and_buffer(frame):
            return
        self._actions.append(slot)
        self.n_pushes += 1
        self._maybe_emit_window()

    # ---- Cached-embedding path: reuse the predator's existing encoder
    # forward instead of paying for a second encode. silent uses this;
    # relay still uses the frame-based path above.

    def push_initial_emb(self, emb: np.ndarray) -> None:
        if not self._buffer_emb(emb):
            return
        self.n_pushes += 1
        self._maybe_emit_window()

    def push_step_emb(self, thrust: np.ndarray, emb: np.ndarray) -> None:
        slot = self._tile_action(thrust)
        if not self._buffer_emb(emb):
            return
        self._actions.append(slot)
        self.n_pushes += 1
        self._maybe_emit_window()

    def _tile_action(self, thrust: np.ndarray) -> np.ndarray:
        thrust = np.asarray(thrust, dtype=np.float32).reshape(-1)
        if thrust.shape != (self.action_dim,):
            raise ValueError(
                f"thrust shape {thrust.shape} != ({self.action_dim},)")
        return np.tile(thrust, self.frameskip).astype(np.float32, copy=False)

    def _buffer_emb(self, emb: np.ndarray) -> bool:
        flat = _normalize_emb_to_1d(emb)
        if flat is None:
            return False
        self._embs.append(flat)
        return True

    def _encode_and_buffer(self, frame: np.ndarray) -> bool:
        try:
            emb = self.encoder([frame])
        except Exception:
            logger.exception("data_tap: encoder failed; dropping frame")
            return False
        if emb is None:
            return False
        return self._buffer_emb(emb)

    def _maybe_emit_window(self) -> None:
        if len(self._embs) < self.window_size:
            return
        if len(self._actions) < self.history:
            return
        emb = np.stack(list(self._embs), axis=0).astype(np.float32, copy=False)
        action_arr = np.stack(list(self._actions), axis=0).astype(np.float32)
        self.n_windows += 1
        self._windows_emitted += 1
        self._batch_buf.append((emb, action_arr))
        if len(self._batch_buf) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        if len(self._batch_buf) < self.batch_size:
            return
        head = self._batch_buf[:self.batch_size]
        self._batch_buf = self._batch_buf[self.batch_size:]
        embs = np.stack([e for e, _ in head], axis=0)
        actions = np.stack([a for _, a in head], axis=0)
        self.n_batches += 1
        if self.ingest_client is not None:
            self.ingest_client.enqueue(embs, actions)

    def flush_partial(self) -> None:
        """Force-emit any pending windows even without a full batch. Pads
        the trailing batch to batch_size by replaying the last sample —
        used at match-end so the tail isn't lost."""
        if not self._batch_buf:
            return
        while len(self._batch_buf) < self.batch_size:
            self._batch_buf.append(self._batch_buf[-1])
        self._flush_batch()

    def stats(self) -> dict:
        return {
            "pushes":  self.n_pushes,
            "windows": self.n_windows,
            "batches": self.n_batches,
            "buffered_embs":   len(self._embs),
            "buffered_actions": len(self._actions),
            "pending_in_batch": len(self._batch_buf),
        }
