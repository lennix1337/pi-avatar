"""Voice layer of the companion: hands-free listening, expressive speech, lip-sync frames.

    voice = Voice(config, on_speech_start=brain.interrupt)
    async for utt in voice.listen(): ...
    await voice.speak("Oi!", "happy", on_mouth=pet.send_mouth)   # once per sentence
    voice.cancel()                                                # barge-in
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..voiceapi import MouthFn, Utterance, VoiceIO
from . import config as _config
from .audio import FRAME, MIC_SR, rms
from .echo import EchoConfig, EchoGate
from .lipsync import LipSyncConfig
from .mic import FRAME_S, Mic, create_mic
from .sink import BufferedSink, SoundDeviceSink, VirtualSink
from .speaker import Speaker
from .stt import SttBackend, create_stt
from .tts import TtsBackend
from .tts import create as create_tts
from .vad import Segment, Segmenter, SegmenterConfig, create_vad

__all__ = ["Utterance", "Voice", "create_voice"]

log = logging.getLogger("companion.voice")


class Voice(VoiceIO):
    """The real voice. `Utterance.started_at/ended_at` are time.time() of the user's speech."""

    def __init__(self, config: dict | None = None, on_speech_start: Callable[[], None] | None = None, *,
                 mic: Mic | None = None, tts: TtsBackend | None = None, stt: SttBackend | None = None,
                 sink: BufferedSink | None = None, vad=None):
        self.cfg = _config.merge(config)
        self.on_speech_start = on_speech_start
        self._mic = mic
        self._tts = tts
        self._stt = stt
        self._sink = sink
        self._vad = vad
        self._speaker: Speaker | None = None
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="voice")
        self._tts_lock = asyncio.Lock()
        self._stt_lock = asyncio.Lock()
        self._segments: asyncio.Queue[tuple[Segment, float] | None] = asyncio.Queue()
        self._out: asyncio.Queue[Utterance | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._ptt_down = False
        self._ptt_events: asyncio.Queue[bool] = asyncio.Queue()
        self._closed = False
        self.gate = EchoGate(EchoConfig.from_dict(self.cfg["echo"]))
        self.stats: dict[str, list[float]] = {"stt_s": [], "end_to_text_s": []}

    # ------------------------------------------------------------------ speak
    async def _ensure_speaker(self) -> Speaker:
        async with self._tts_lock:
            if self._speaker is None:
                loop = asyncio.get_running_loop()
                if self._tts is None:
                    t = self.cfg["tts"]
                    self._tts = create_tts(t["backend"], **t.get("options", {}))
                await loop.run_in_executor(self._pool, self._tts.load)
                if self._sink is None:
                    out = self.cfg["output"]
                    if out.get("backend") == "virtual":
                        self._sink = VirtualSink(self._tts.sample_rate, speed=float(out.get("speed", 1.0)))
                    else:
                        self._sink = SoundDeviceSink(self._tts.sample_rate, out.get("device"), out.get("block_ms", 20))
                lip = self.cfg["lipsync"]
                self._speaker = Speaker(self._tts, self._sink, LipSyncConfig(fps=lip["fps"], gain=lip["gain"]))
            return self._speaker

    async def speak(self, text: str, emotion: str, on_mouth: MouthFn) -> None:
        """Play one sentence; returns when it has been heard. Queue several calls to pipeline them."""
        sp = await self._ensure_speaker()
        await sp.speak(text, emotion, on_mouth)

    def cancel(self) -> None:
        """Barge-in: stop playback now and drop queued speech."""
        if self._speaker is not None:
            self._speaker.cancel()

    @property
    def speaking(self) -> bool:
        return self._speaker is not None and self._speaker.speaking

    # ------------------------------------------------------------------ listen
    async def listen(self) -> AsyncIterator[Utterance]:
        """Yield one Utterance per turn of the user, forever (or until the mic source ends)."""
        if not self._tasks:
            self._tasks = [asyncio.create_task(self._capture_loop()), asyncio.create_task(self._stt_loop())]
        while True:
            utt = await self._out.get()
            if utt is None:
                return
            yield utt

    def ptt(self, down: bool) -> None:
        """Push-to-talk key state (only used when listen.mode == "ptt")."""
        self._ptt_events.put_nowait(bool(down))

    async def _ensure_stt(self) -> SttBackend:
        async with self._stt_lock:
            if self._stt is None:
                self._stt = create_stt(self.cfg["stt"])
                await asyncio.get_running_loop().run_in_executor(self._pool, self._stt.load)
            return self._stt

    async def start(self) -> None:
        await self.warmup()

    async def warmup(self) -> None:
        """Load the models now instead of at first use."""
        await asyncio.gather(self._ensure_speaker(), self._ensure_stt())

    def _fire_start(self) -> None:
        if self.cfg["listen"].get("barge_in_cancel", True) and self.speaking:
            self.cancel()
        if self.on_speech_start is not None:
            try:
                self.on_speech_start()
            except Exception:
                log.exception("on_speech_start failed")

    async def _capture_loop(self) -> None:
        lc = self.cfg["listen"]
        mic = self._mic or create_mic(lc.get("source", "device"), lc.get("input_device"), bool(lc.get("realtime", False)))
        self._mic = mic
        vad = self._vad or await asyncio.get_running_loop().run_in_executor(
            self._pool, create_vad, lc["vad"].get("backend", "silero"))
        seg = Segmenter(SegmenterConfig.from_dict(lc["vad"]))
        ptt = lc.get("mode") == "ptt"
        threshold = lc["vad"].get("threshold", 0.5)
        k = 0
        try:
            async for frame in mic.frames():
                if frame.size != FRAME:
                    continue
                k += 1
                t = k * FRAME_S                    # stream clock, end of this frame
                events = []
                while ptt and not self._ptt_events.empty():
                    down = self._ptt_events.get_nowait()
                    if down and not self._ptt_down:
                        seg.begin(t - FRAME_S)
                        events.append(("start", t - FRAME_S))
                    elif not down and self._ptt_down:
                        events += seg.flush(t)
                    self._ptt_down = down
                if ptt:
                    if self._ptt_down:
                        seg.add(frame)
                else:
                    ref = self._speaker_ref()
                    self.gate.note_playback(t, self.speaking)
                    ok = self.gate.allow(rms(frame), ref)
                    p = vad(frame) if ok else 0.0
                    events += seg.push(frame, p, t, self.gate.vad_threshold(threshold))
                for kind, val in events:
                    if kind == "start":
                        self._fire_start()
                    elif kind == "end":
                        await self._segments.put((val, mic.t0))
        finally:
            for kind, val in seg.flush(k * FRAME_S):
                if kind == "end":
                    await self._segments.put((val, mic.t0))
            await self._segments.put(None)

    def _speaker_ref(self) -> float:
        return self._sink.recent_rms() if self._sink is not None and self._speaker is not None else 0.0

    async def _stt_loop(self) -> None:
        loop = asyncio.get_running_loop()
        min_chars = self.cfg["stt"].get("min_chars", 2)
        while True:
            item = await self._segments.get()
            if item is None:
                await self._out.put(None)
                return
            segment, t0 = item
            stt = await self._ensure_stt()
            t = time.perf_counter()
            try:
                text = await loop.run_in_executor(self._pool, stt.transcribe, segment.audio.astype(np.float32))
            except Exception:
                log.exception("stt failed")
                continue
            dt = time.perf_counter() - t
            self.stats["stt_s"].append(dt)
            if len(text.strip()) >= min_chars:
                await self._out.put(Utterance(text.strip(), t0 + segment.t_start, t0 + segment.t_end))

    # ------------------------------------------------------------------ close
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        if self._mic is not None:
            await self._mic.close()
        if self._speaker is not None:
            await self._speaker.close()
        for b in (self._tts, self._stt):
            if b is not None:
                b.close()
        self._pool.shutdown(wait=False, cancel_futures=True)
        await self._out.put(None)


def create_voice(config: dict | None = None, on_speech_start: Callable[[], None] | None = None) -> Voice:
    """The real voice with the caller's config merged over the defaults (`start()` loads its models)."""
    return Voice(config, on_speech_start=on_speech_start)
