from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from .base import TtsBackend, style_for, to_float32


class SupertonicTts(TtsBackend):
    """Supertonic 3 (ONNX, OpenRAIL). Preset voices; emotion = pace."""

    name = "supertonic"
    sample_rate = 44100

    def __init__(self, voice: str = "F1", total_steps: int = 8, **_):
        self.voice = voice
        self.total_steps = total_steps
        self.tts = None
        self.style = None

    def load(self) -> None:
        from supertonic import TTS

        self.tts = TTS(auto_download=True)
        self.style = self.tts.get_voice_style(voice_name=self.voice)
        self.sample_rate = int(getattr(self.tts, "sample_rate", self.sample_rate))

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        st = style_for(emotion)
        try:
            wav, _dur = self.tts.synthesize(text, voice_style=self.style, lang="pt",
                                            speed=st.speed, total_steps=self.total_steps)
        except TypeError:
            wav, _dur = self.tts.synthesize(text, voice_style=self.style, lang="pt")
        yield to_float32(wav)
