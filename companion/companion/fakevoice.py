"""Voz de mentira para rodar sem áudio: o texto digitado vira fala dele; a fala dela é impressa."""

from __future__ import annotations

import asyncio
import math
import sys
import time
from typing import AsyncIterator

from .voiceapi import MouthFn, Utterance, VoiceIO

CPS = 16.0  # caracteres por segundo da "fala" simulada


class FakeVoice(VoiceIO):
    def __init__(self, realtime: bool = True, out=None):
        self.realtime = realtime
        self.out = out if out is not None else sys.stdout
        self._queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        self._speaking: set[asyncio.Task] = set()   # falas em curso ou na fila (o orquestrador enfileira até duas)
        self._lock = asyncio.Lock()                  # uma fala por vez, como a voz real
        self.spoken: list[tuple[str, str]] = []

    def feed(self, text: str) -> None:
        """Entrada digitada: avisa o início da fala (barge-in) e entrega o texto."""
        text = text.strip()
        if not text:
            return
        if self.on_speech_start:
            self.on_speech_start()
        now = time.monotonic()
        self._queue.put_nowait(Utterance(text, now, now))

    def finish(self) -> None:
        self._queue.put_nowait(None)

    async def listen(self) -> AsyncIterator[Utterance]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def speak(self, text: str, emotion: str, on_mouth: MouthFn) -> None:
        self.spoken.append((text, emotion))
        self._speaking.add(asyncio.current_task())
        try:
            async with self._lock:
                print(f"  ela [{emotion}]> {text}", file=self.out, flush=True)
                seconds = len(text) / CPS if self.realtime else 0.0
                try:
                    for i in range(int(seconds * 30)):
                        phase = i / 30
                        on_mouth({"t": "mouth", "aa": round(abs(math.sin(phase * 9)) * 0.7, 3), "ih": 0.0,
                                  "ou": round(abs(math.sin(phase * 5)) * 0.2, 3), "ee": 0.0, "oh": 0.0})
                        await asyncio.sleep(1 / 30)
                finally:
                    on_mouth({"t": "mouth", "aa": 0.0, "ih": 0.0, "ou": 0.0, "ee": 0.0, "oh": 0.0})
        finally:
            self._speaking.discard(asyncio.current_task())

    def cancel(self) -> None:
        for task in list(self._speaking):
            if task is not asyncio.current_task():
                task.cancel()
