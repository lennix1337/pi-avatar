"""Text-to-speech backends behind one streaming interface."""

from __future__ import annotations

from .base import EMOTIONS, EmotionStyle, TtsBackend, style_for

__all__ = ["EMOTIONS", "EmotionStyle", "TtsBackend", "style_for", "create"]


def create(name: str, **options) -> TtsBackend:
    """Build a backend by name; heavy imports happen inside `load()`."""
    if name == "kokoro":
        from .kokoro_tts import KokoroTts as cls
    elif name == "supertonic":
        from .supertonic_tts import SupertonicTts as cls
    elif name == "voxcpm":
        from .voxcpm_tts import VoxCpmTts as cls
    elif name == "omnivoice":
        from .omnivoice_tts import OmniVoiceTts as cls
    elif name == "qwen3":
        from .qwen3_tts import Qwen3Tts as cls
    elif name == "fake":
        from .fake import FakeTts as cls
    else:
        raise ValueError(f"unknown TTS backend {name!r}")
    return cls(**options)
