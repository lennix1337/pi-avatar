"""Audio output sinks. All of them expose one timeline: seconds of audio written/played."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

import numpy as np


class BufferedSink:
    """FIFO of float32 samples drained by a clock; subclasses decide who drives the drain."""

    def __init__(self, sample_rate: int, latency_s: float = 0.0):
        self.sample_rate = sample_rate
        self.latency_s = latency_s            # device latency between "taken" and audible
        self._lock = threading.Lock()
        self._buf: deque[np.ndarray] = deque()
        self._buffered = 0                    # samples waiting
        self._taken = 0                       # samples handed to the device
        self._written = 0                     # samples ever accepted (timeline end)
        self._rms_log: deque[tuple[int, float]] = deque(maxlen=400)  # (end sample, rms) of drained blocks
        self._scheduled: deque[tuple[float, int, int]] = deque()  # (DAC start, first sample, end sample)
        self._played = 0
        self._echo: deque[tuple[float, np.ndarray]] = deque()
        self._echo_at = float("-inf")

    # -- producer side ------------------------------------------------
    def write(self, samples: np.ndarray) -> float:
        """Queue audio; returns its offset (s) on the timeline."""
        x = np.ascontiguousarray(samples, dtype=np.float32)
        with self._lock:
            off = self._written / self.sample_rate
            self._buf.append(x)
            self._buffered += x.size
            self._written += x.size
        return off

    def clear(self) -> None:
        """Drop everything not yet taken by the device (barge-in)."""
        with self._lock:
            self._buf.clear()
            self._buffered = 0
            self._written = self._taken

    # -- consumer side --------------------------------------------------
    def take(self, n: int, *, dac_start: float | None = None) -> np.ndarray:
        """Take up to n samples (zero-padded to n) and record when the real PCM will play."""
        out = np.zeros(n, np.float32)
        with self._lock:
            got = 0
            while got < n and self._buf:
                head = self._buf[0]
                m = min(n - got, head.size)
                out[got:got + m] = head[:m]
                if m == head.size:
                    self._buf.popleft()
                else:
                    self._buf[0] = head[m:]
                got += m
            self._buffered -= got
            if got and dac_start is not None:
                self._scheduled.append((dac_start, self._taken, self._taken + got))
            self._taken += got
            now = time.monotonic()
            if got:
                self._rms_log.append((self._taken, float(np.sqrt(np.mean(out[:got] ** 2)))))
                self._echo_at = now
            if now - self._echo_at < 0.5:
                self._echo.append((now, out))
            while self._echo and now - self._echo[0][0] >= 0.5:
                self._echo.popleft()
        return out

    # -- observers --------------------------------------------------------
    @property
    def buffered_s(self) -> float:
        return self._buffered / self.sample_rate

    @property
    def written_s(self) -> float:
        return self._written / self.sample_rate

    def position(self) -> float:
        """Seconds of the timeline that are audible *now* (device latency removed)."""
        return max(0.0, self._taken / self.sample_rate - self.latency_s)

    def playback(self) -> tuple[float, bool]:
        """Audio timeline position and whether it points to PCM, not a padding gap."""
        return self.position(), True

    def recent_rms(self, window_s: float = 0.3) -> float:
        """Loudest drained block over the last `window_s`: the echo reference."""
        lo = self._taken - int(window_s * self.sample_rate)
        with self._lock:
            vals = [r for end, r in self._rms_log if end >= lo]
        return max(vals) if vals else 0.0

    def echo_reference(self, window_s: float) -> np.ndarray | None:
        """Recent PCM handed to the DAC, including padding needed to align delayed echo."""
        cutoff = time.monotonic() - window_s
        with self._lock:
            if self._echo_at < cutoff:
                return None
            blocks = [x for t, x in self._echo if t >= cutoff]
        return np.concatenate(blocks)[-int(window_s * self.sample_rate):] if blocks else None

    async def prepare(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class VirtualSink(BufferedSink):
    """A fake sound card that drains in (scaled) real time. For tests and file rendering."""

    def __init__(self, sample_rate: int, latency_s: float = 0.0, speed: float = 1.0, block_s: float = 0.02):
        super().__init__(sample_rate, latency_s)
        self.speed = speed
        self._t = time.monotonic()
        self._block = int(block_s * sample_rate)
        self.starved_s = 0.0      # silence played after the first sample (underruns, not idle time before it)
        self._started = False

    def _drain(self) -> None:
        now = time.monotonic()
        want = int((now - self._t) * self.sample_rate * self.speed)
        while want >= self._block:
            if self._buffered == 0:
                # nothing to play: the device idles, wall time passes
                if self._started:
                    self.starved_s += want / (self.sample_rate * self.speed)
                self._t = now
                return
            self._started = True
            self.take(min(self._block, self._buffered))
            want -= self._block
            self._t += self._block / (self.sample_rate * self.speed)
        # keep the remainder for the next call

    def position(self) -> float:
        self._drain()
        return super().position()

    def recent_rms(self, window_s: float = 0.3) -> float:
        self._drain()
        return super().recent_rms(window_s)

    def echo_reference(self, window_s: float) -> np.ndarray | None:
        self._drain()
        return super().echo_reference(window_s)

    def write(self, samples: np.ndarray) -> float:
        if self._buffered == 0:
            self._drain()
            self._t = time.monotonic()
        return super().write(samples)


class SoundDeviceSink(BufferedSink):
    """Real output through PortAudio (WASAPI shared on Windows)."""

    def __init__(self, sample_rate: int, device=None, block_ms: int = 20):
        super().__init__(sample_rate)
        import sounddevice as sd

        self._device_task: asyncio.Task | None = None
        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32", device=device,
            blocksize=int(sample_rate * block_ms / 1000), callback=self._callback, latency="low")
        self.latency_s = float(self._stream.latency)

    def _callback(self, outdata, frames, timing, _status) -> None:
        outdata[:, 0] = self.take(frames, dac_start=timing.outputBufferDacTime)

    def playback(self) -> tuple[float, bool]:
        # PortAudio reports both times on the same clock. Only the actual PCM advances
        # this timeline: the zero padding of an underrun must not consume future visemes.
        now = self._stream.time
        with self._lock:
            while self._scheduled:
                dac_start, start, end = self._scheduled[0]
                if now < dac_start:
                    break
                if now < dac_start + (end - start) / self.sample_rate:
                    return start / self.sample_rate + now - dac_start, True
                self._played = end
                self._scheduled.popleft()
            return self._played / self.sample_rate, False

    def position(self) -> float:
        return self.playback()[0]

    def clear(self) -> None:
        super().clear()
        with self._lock:
            pending_dac = bool(self._scheduled)
            self._scheduled.clear()
            self._played = self._taken
        if pending_dac:
            self._device_task = asyncio.create_task(asyncio.to_thread(self._abort_and_start))

    def _abort_and_start(self) -> None:
        self._stream.abort()
        self._stream.start()

    async def prepare(self) -> None:
        # Keep device operations alive across cancellation of an individual sentence.
        if self._device_task is None:
            self._device_task = asyncio.create_task(asyncio.to_thread(self._stream.start))
        await asyncio.shield(self._device_task)

    async def shutdown(self) -> None:
        try:
            if self._device_task is not None:
                await asyncio.shield(self._device_task)
        finally:
            await asyncio.to_thread(self._stream.close)


def list_devices() -> list[dict]:
    """Enumerate audio devices (does not open any stream)."""
    import sounddevice as sd

    return [dict(d, index=i) for i, d in enumerate(sd.query_devices())]
