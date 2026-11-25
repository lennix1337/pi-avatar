from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from .base import PERSONA, TtsBackend, style_for, to_float32


class Qwen3Tts(TtsBackend):
    """Qwen3-TTS 1.7B VoiceDesign (Apache-2.0): persona and emotion are one instruction."""

    name = "qwen3"
    sample_rate = 24000

    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", **_):
        self.model_id = model_id
        self.model = None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        st = style_for(emotion)
        wavs, sr = self.model.generate_voice_design(
            text=text, language="Portuguese", instruct=f"{PERSONA}; {st.desc}")
        self.sample_rate = int(sr)
        yield to_float32(wavs[0])
