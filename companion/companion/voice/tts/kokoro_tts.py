from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from .base import TtsBackend, style_for, to_float32


class KokoroTts(TtsBackend):
    """Kokoro-82M (Apache-2.0), Brazilian Portuguese voice. Emotion = pace only."""

    name = "kokoro"
    sample_rate = 24000

    def __init__(self, voice: str = "pf_dora", device: str | None = None, **_):
        self.voice = voice
        self.device = device
        self.pipe = None

    def load(self) -> None:
        from kokoro import KPipeline

        self.pipe = KPipeline(lang_code="p", device=self.device)

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        st = style_for(emotion)
        for _gs, _ps, audio in self.pipe(text, voice=self.voice, speed=st.speed):
            if cancel is not None and cancel.is_set():
                return
            if audio is not None:
                yield to_float32(audio)
