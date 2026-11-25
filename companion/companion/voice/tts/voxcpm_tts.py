from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .base import PERSONA, TtsBackend, style_for, to_float32

PERSONA_LINE = "Oi! Eu sou a sua companheira. Vamos conversar um pouquinho hoje?"


def cache_dir() -> Path:
    base = os.environ.get("COMPANION_VOICE_CACHE") or str(Path.home() / ".cache" / "companion-voice")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


class VoxCpmTts(TtsBackend):
    """VoxCPM2 (Apache-2.0). The persona is *designed* from a text description once,
    then reused as a reference clip so every sentence keeps the same timbre. Emotion is a
    style prompt in parentheses."""

    name = "voxcpm"
    sample_rate = 48000

    def __init__(self, steps: int = 8, cfg: float = 2.0, **_):
        self.steps = steps
        self.cfg = cfg
        self.model = None
        self.ref: str | None = None

    def load(self) -> None:
        import soundfile as sf
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        self.sample_rate = int(self.model.tts_model.sample_rate)
        ref = cache_dir() / "persona_voxcpm.wav"
        if not ref.exists():
            wav = self.model.generate(
                text=f"({PERSONA}){PERSONA_LINE}", cfg_value=self.cfg, inference_timesteps=10)
            sf.write(str(ref), to_float32(wav), self.sample_rate)
        self.ref = str(ref)

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        st = style_for(emotion)
        prompt = f"({st.desc})" + text
        for chunk in self.model.generate_streaming(
            text=prompt, reference_wav_path=self.ref, cfg_value=self.cfg, inference_timesteps=self.steps
        ):
            if cancel is not None and cancel.is_set():
                return
            yield to_float32(chunk)
