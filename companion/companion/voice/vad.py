"""Voice activity detection: per-frame probability + the turn segmenter state machine."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import FRAME, MIC_SR, dbfs, rms

FRAME_S = FRAME / MIC_SR


class EnergyVad:
    """Dependency-free fallback: loudness over an adaptive noise floor."""

    def __init__(self, margin_db: float = 12.0, floor_dbfs: float = -55.0):
        self.margin_db = margin_db
        self.noise = floor_dbfs
        self.floor = floor_dbfs

    def reset(self) -> None:
        self.noise = self.floor

    def __call__(self, frame: np.ndarray) -> float:
        db = dbfs(rms(frame))
        if db < self.noise + self.margin_db:  # only quiet frames teach the noise floor
            self.noise += 0.05 * (max(db, self.floor) - self.noise)
        return 1.0 / (1.0 + math.exp(-(db - (self.noise + self.margin_db)) / 2.5))


class SileroVad:
    """Silero VAD v5 through its ONNX build (CPU, ~0.3 ms per 32 ms frame)."""

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self.model = load_silero_vad(onnx=True)

    def reset(self) -> None:
        self.model.reset_states()

    def __call__(self, frame: np.ndarray) -> float:
        t = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        return float(self.model(t, MIC_SR).item())


def create_vad(name: str):
    if name == "silero":
        try:
            return SileroVad()
        except Exception:  # not installed or model missing: degrade to energy instead of dying
            return EnergyVad()
    return EnergyVad()


@dataclass
class Segment:
    audio: np.ndarray       # 16 kHz mono, includes the pre-roll
    t_start: float          # clock seconds when speech began (not counting pre-roll)
    t_end: float            # clock seconds when the last speech frame ended


@dataclass
class SegmenterConfig:
    threshold: float = 0.5
    neg_threshold: float = 0.35
    start_ms: float = 96
    min_speech_ms: float = 250
    end_silence_ms: float = 550
    preroll_ms: float = 320
    max_utterance_s: float = 30.0

    @classmethod
    def from_dict(cls, d: dict) -> SegmenterConfig:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class Segmenter:
    """Frames + speech probabilities in, turns out.

    `push()` returns events: ("start", t) once speech is confirmed (barge-in hook) and
    ("end", Segment) when the speaker stops, or ("drop", None) for a too-short burst.
    """

    def __init__(self, cfg: SegmenterConfig | None = None):
        self.cfg = cfg or SegmenterConfig()
        self.reset()

    def reset(self) -> None:
        c = self.cfg
        self._pre: deque[np.ndarray] = deque(maxlen=max(1, round(c.preroll_ms / 1000 / FRAME_S)))
        self._frames: list[np.ndarray] = []
        self._state = "idle"        # idle -> pending -> speech
        self._run = 0               # consecutive speech frames (idle/pending)
        self._silence = 0           # consecutive non-speech frames (speech)
        self._speech_frames = 0
        self._t_start = 0.0
        self._t_last_speech = 0.0

    @property
    def in_turn(self) -> bool:
        return self._state == "speech"

    def begin(self, t: float) -> None:
        """Open a turn by hand (push-to-talk press) at clock time `t`."""
        self.reset()
        self._state = "speech"
        self._t_start = t
        self._t_last_speech = t

    def add(self, frame: np.ndarray) -> None:
        """Append a frame to a manually opened turn."""
        self._frames.append(frame)
        self._speech_frames += 1

    def flush(self, t: float) -> list[tuple[str, object]]:
        """Force-close a turn in progress (push-to-talk release, end of input)."""
        if self._state == "idle" or not self._frames:
            self.reset()
            return []
        ev = self._close(t)
        self.reset()
        return [ev]

    def _close(self, t_end: float):
        c = self.cfg
        speech_s = self._speech_frames * FRAME_S
        if speech_s * 1000 < c.min_speech_ms:
            return ("drop", None)
        return ("end", Segment(np.concatenate(self._frames), self._t_start, t_end))

    def push(self, frame: np.ndarray, prob: float, t: float, threshold: float | None = None) -> list[tuple[str, object]]:
        """`t` is the clock time at the *end* of this frame."""
        c = self.cfg
        thr = c.threshold if threshold is None else threshold
        events: list[tuple[str, object]] = []
        is_speech = prob >= (c.neg_threshold if self._state == "speech" else thr)
        if self._state in ("idle", "pending"):
            if is_speech:
                if self._state == "idle":
                    self._frames = list(self._pre)
                    self._t_start = t - FRAME_S
                    self._speech_frames = 0
                self._state = "pending"
                self._frames.append(frame)
                self._run += 1
                self._speech_frames += 1
                if self._run * FRAME_S * 1000 >= c.start_ms:
                    self._state = "speech"
                    self._silence = 0
                    self._t_last_speech = t
                    events.append(("start", self._t_start))
            else:
                self._run = 0
                self._state = "idle"
                self._frames = []
                self._pre.append(frame)
            return events
        # in speech
        self._frames.append(frame)
        if is_speech:
            self._silence = 0
            self._speech_frames += 1
            self._t_last_speech = t
        else:
            self._silence += 1
        ended = self._silence * FRAME_S * 1000 >= c.end_silence_ms
        too_long = len(self._frames) * FRAME_S >= c.max_utterance_s
        if ended or too_long:
            events.append(self._close(self._t_last_speech))
            self.reset()
        return events
