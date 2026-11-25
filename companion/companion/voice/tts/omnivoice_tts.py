from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from .base import TtsBackend, style_for, to_float32

TAGS = {"happy": "", "excited": "", "surprised": "[surprise-ah] ", "embarrassed": "", "teasing": "", "smug": ""}


class OmniVoiceTts(TtsBackend):
    """OmniVoice (weights CC-BY-NC). Voice design by attributes; emotion = pace and tags."""

    name = "omnivoice"
    sample_rate = 24000

    def __init__(self, instruct: str = "female, young, high pitch", steps: int = 16, **_):
        self.instruct = instruct
        self.steps = steps
        self.model = None

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

    def stream(self, text: str, emotion: str, cancel: threading.Event | None = None) -> Iterator[np.ndarray]:
        st = style_for(emotion)
        audio = self.model.generate(text=TAGS.get(emotion, "") + text, instruct=self.instruct,
                                    num_step=self.steps, speed=st.speed)
        yield to_float32(audio[0])
