from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmotionStyle:
    """How one emotion changes delivery. Each backend uses what its model allows."""

    desc: str               # English style phrase for prompt-controlled models
    speed: float = 1.0      # pace factor
    exaggeration: float = 0.5   # emotion intensity for models that expose it (0..1)
    temperature: float = 0.8


EMOTIONS: dict[str, EmotionStyle] = {
    "neutral": EmotionStyle("natural, relaxed", 1.0, 0.45),
    "happy": EmotionStyle("cheerful, smiling, bright and upbeat", 1.06, 0.7),
    "excited": EmotionStyle("very excited, energetic, fast and high-pitched", 1.15, 0.9, 0.9),
    "angry": EmotionStyle("annoyed, sharp and firm", 1.08, 0.85),
    "sad": EmotionStyle("sad, soft, slow and gentle", 0.9, 0.55),
    "relaxed": EmotionStyle("calm, warm and unhurried", 0.95, 0.4),
    "surprised": EmotionStyle("surprised, astonished, rising intonation", 1.08, 0.85),
    "smug": EmotionStyle("smug, playful, teasing", 1.0, 0.7),
    "teasing": EmotionStyle("teasing, playful and mischievous", 1.03, 0.75),
    "embarrassed": EmotionStyle("embarrassed, shy, a little hesitant", 0.95, 0.6),
    "bored": EmotionStyle("bored, flat and lazy", 0.92, 0.35),
    "confused": EmotionStyle("confused, unsure, questioning", 0.97, 0.6),
}

# One persona for every backend: designed, not cloned from any real person.
PERSONA = "a young Brazilian woman in her early twenties, bright, sweet and cute voice, lively and expressive"


def style_for(emotion: str) -> EmotionStyle:
    return EMOTIONS.get(emotion, EMOTIONS["neutral"])


class TtsBackend:
    """Streaming TTS: `stream()` yields float32 mono chunks at `sample_rate`."""

    name = "base"
    sample_rate = 24000

    def load(self) -> None:  # pragma: no cover - heavy
        pass

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def synth(self, text: str, emotion: str = "neutral") -> np.ndarray:
        chunks = [c for c in self.stream(text, emotion) if c.size]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def close(self) -> None:
        pass


def to_float32(x) -> np.ndarray:
    """Torch tensor / list / ndarray -> flat float32 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)
