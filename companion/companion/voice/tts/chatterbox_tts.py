from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from .base import TtsBackend, style_for, to_float32


class ChatterboxTts(TtsBackend):
    """Chatterbox Multilingual (MIT). `exaggeration` carries the emotion."""

    name = "chatterbox"
    sample_rate = 24000

    def __init__(self, ref_wav: str | None = None, cfg_weight: float = 0.4, **_):
        self.ref_wav = ref_wav
        self.cfg_weight = cfg_weight
        self.model = None

    def load(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self.sample_rate = int(self.model.sr)

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        st = style_for(emotion)
        wav = self.model.generate(
            text, language_id="pt", audio_prompt_path=self.ref_wav,
            exaggeration=st.exaggeration, cfg_weight=self.cfg_weight, temperature=st.temperature)
        yield to_float32(wav)
