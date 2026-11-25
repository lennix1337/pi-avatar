from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from .base import TtsBackend


class FakeTts(TtsBackend):
    """Deterministic stand-in for tests: a vowel-ish tone per word, optional delay."""

    name = "fake"

    def __init__(self, sample_rate: int = 16000, delay: float = 0.0, secs_per_word: float = 0.2,
                 chunk_secs: float = 0.1, **_):
        self.sample_rate = sample_rate
        self.delay = delay
        self.secs_per_word = secs_per_word
        self.chunk_secs = chunk_secs
        self.calls: list[tuple[str, str]] = []

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        self.calls.append((text, emotion))
        if self.delay:
            time.sleep(self.delay)
        total = max(1, len(text.split())) * self.secs_per_word
        n = int(total * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        step = int(self.chunk_secs * self.sample_rate)
        for i in range(0, n, step):
            if cancel is not None and cancel.is_set():
                return
            yield audio[i:i + step]
