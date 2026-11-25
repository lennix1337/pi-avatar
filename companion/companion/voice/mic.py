"""Microphone sources: async iterators of 16 kHz mono frames of `FRAME` samples."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

from .audio import FRAME, MIC_SR, read_wav_16k

FRAME_S = FRAME / MIC_SR


class Mic:
    """Base class. `t0` is the wall-clock (time.time) instant of the first frame."""

    t0: float = 0.0

    def frames(self) -> AsyncIterator[np.ndarray]:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class ArrayMic(Mic):
    """Plays an array as if it were a microphone. `realtime=False` runs as fast as the consumer pulls."""

    def __init__(self, audio: np.ndarray, realtime: bool = False, tail_s: float = 1.5, loop_forever: bool = False):
        pad = np.zeros(int(tail_s * MIC_SR), np.float32)
        self.audio = np.concatenate([np.asarray(audio, np.float32), pad])
        self.realtime = realtime
        self.loop_forever = loop_forever
        self.t0 = time.time()
        self._closed = False

    async def frames(self) -> AsyncIterator[np.ndarray]:
        n = self.audio.size // FRAME
        k = 0
        start = time.monotonic()
        while not self._closed:
            if k >= n:
                if not self.loop_forever:
                    return
                k = 0
            if self.realtime:
                await asyncio.sleep(max(0.0, start + (k + 1) * FRAME_S - time.monotonic()))
            else:
                await asyncio.sleep(0)
            yield self.audio[k * FRAME:(k + 1) * FRAME]
            k += 1

    async def close(self) -> None:
        self._closed = True


class FileMic(ArrayMic):
    def __init__(self, path: str, **kw):
        super().__init__(read_wav_16k(path), **kw)


class SoundDeviceMic(Mic):
    """The real microphone through PortAudio. Opening it starts recording: only on explicit use."""

    def __init__(self, device=None):
        self.device = device
        self._q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(self, indata, frames, _t, _status) -> None:
        x = indata[:, 0].copy()
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._put, x)

    def _put(self, x: np.ndarray) -> None:
        if self._q.full():
            self._q.get_nowait()   # drop the oldest rather than build latency
        self._q.put_nowait(x)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", device=self.device,
                                      blocksize=FRAME, callback=self._callback)
        self.t0 = time.time()
        self._stream.start()
        while True:
            yield await self._q.get()

    async def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def create_mic(source: str, device=None, realtime: bool = False) -> Mic:
    if source.startswith("file:"):
        return FileMic(source[5:], realtime=realtime)
    if source == "device":
        return SoundDeviceMic(device)
    raise ValueError(f"unknown listen.source {source!r}")
