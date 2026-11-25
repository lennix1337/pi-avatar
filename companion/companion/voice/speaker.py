"""Sentence-pipelined speech.

One worker thread synthesizes sentences strictly in order and appends the audio to the sink as
it is produced, so sentence N+1 is synthesized while N is still playing and its first samples
land in the buffer before N runs out. A single 30 Hz ticker reads the sink's playback clock
and sends the mouth frame of whichever sentence is audible right now.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .lipsync import LipSyncConfig, StreamingLipSync, resample, zero_mouth
from .sink import BufferedSink
from .tts.base import TtsBackend

log = logging.getLogger("companion.voice")

MouthCb = Callable[[dict], None]


@dataclass
class _Item:
    text: str
    emotion: str
    on_mouth: MouthCb
    done: asyncio.Future
    lip: StreamingLipSync
    start: float | None = None        # offset on the sink timeline of its first sample
    end: float = 0.0                  # offset of the end of what has been written
    synth_done: bool = False
    error: Exception | None = None
    t_submit: float = field(default_factory=time.perf_counter)
    first_audio_s: float | None = None   # submit -> first chunk written


class Speaker:
    def __init__(self, backend: TtsBackend, sink: BufferedSink, lip_cfg: LipSyncConfig | None = None):
        self.backend = backend
        self.sink = sink
        self.lip_cfg = lip_cfg or LipSyncConfig()
        self._lock = threading.Lock()         # guards queue/items/cancel against the worker thread
        self._jobs: queue.Queue[_Item | None] = queue.Queue()
        self._items: list[_Item] = []         # accepted, not finished (playback order)
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._ticker: asyncio.Task | None = None
        self._closed = False
        self.last_end = 0.0                   # time.monotonic() of the last audible sample (echo tail)
        self.history: list[_Item] = []        # most recent finished items, for timing inspection

    # -- public ---------------------------------------------------------
    @property
    def speaking(self) -> bool:
        return bool(self._items) or self.sink.buffered_s > 0

    async def speak(self, text: str, emotion: str, on_mouth: MouthCb) -> None:
        text = text.strip()
        if not text or self._closed:
            return
        loop = asyncio.get_running_loop()
        item = _Item(text, emotion, on_mouth, loop.create_future(),
                     StreamingLipSync(self.backend.sample_rate, self.lip_cfg))
        with self._lock:
            self._items.append(item)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="tts", daemon=True)
                self._worker.start()
            self._jobs.put(item)
        if self._ticker is None or self._ticker.done():
            self.sink.start()
            self._ticker = loop.create_task(self._tick())
        await item.done
        if item.error is not None:
            raise item.error

    def cancel(self) -> None:
        """Stop now: silence the sink, abort the synthesis in flight, drop everything queued."""
        with self._lock:
            self._cancel.set()
            self.sink.clear()
            dropped, self._items = self._items, []
            while True:
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    break
        for i in dropped:
            if not i.done.done():
                i.done.set_result(None)
        for cb in {id(i.on_mouth): i.on_mouth for i in dropped}.values():
            self._emit(cb, zero_mouth())
        if dropped:
            self.last_end = time.monotonic()

    async def close(self) -> None:
        self._closed = True
        self.cancel()
        self._jobs.put(None)
        if self._ticker is not None:
            self._ticker.cancel()
            try:
                await self._ticker
            except asyncio.CancelledError:
                pass
        self.sink.close()

    # -- worker thread ----------------------------------------------------
    def _work(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                return
            with self._lock:
                if item not in self._items:   # cancelled while queued
                    continue
                if self._cancel.is_set():
                    self._cancel = threading.Event()   # fresh flag for work submitted after the cancel
                cancel = self._cancel
            try:
                for chunk in self.backend.stream(item.text, item.emotion, cancel):
                    if chunk.size == 0:
                        continue
                    x = chunk if self.backend.sample_rate == self.sink.sample_rate else resample(
                        chunk, self.backend.sample_rate, self.sink.sample_rate)
                    with self._lock:
                        if cancel.is_set() or item not in self._items:
                            break
                        off = self.sink.write(x)
                        if item.start is None:
                            item.start = off
                            item.first_audio_s = time.perf_counter() - item.t_submit
                        item.end = off + x.size / self.sink.sample_rate
                    item.lip.feed(chunk)
            except Exception as e:  # handed to the awaiting speak()
                item.error = e
                log.exception("tts failed for %r", item.text[:40])
            item.lip.feed(final=True)
            item.synth_done = True

    # -- playback clock ---------------------------------------------------
    @staticmethod
    def _emit(cb: MouthCb, weights: dict) -> None:
        try:
            cb(weights)
        except Exception:  # a broken consumer must not stop the voice
            log.exception("on_mouth callback failed")

    async def _tick(self) -> None:
        period = 1.0 / self.lip_cfg.fps
        nxt = time.monotonic()
        last_cb: MouthCb | None = None
        while True:
            nxt += period
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))
            pos = self.sink.position()
            with self._lock:
                items = list(self._items)
            if not items:
                if last_cb is not None:
                    self._emit(last_cb, zero_mouth())
                return
            for it in items:
                if it.synth_done and (it.start is None or pos >= it.end - 1e-6):
                    self._finish(it)
                    continue
                if it.start is not None and pos >= it.start:
                    self._emit(it.on_mouth, it.lip.frame_at(pos - it.start))
                    last_cb = it.on_mouth
                break

    def _finish(self, item: _Item) -> None:
        with self._lock:
            if item in self._items:
                self._items.remove(item)
        self.last_end = time.monotonic()
        self.history = (self.history + [item])[-50:]
        if not item.done.done():
            item.done.set_result(None)
