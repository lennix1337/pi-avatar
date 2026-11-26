"""Remove correlated playback from mic PCM; fall back to a learned energy gate without PCM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import MIC_SR, dbfs, rms
from .lipsync import resample

HOLD_FRAMES = 10   # once the user is recognised, stay open through syllable-level dips (~320 ms)


@dataclass
class EchoConfig:
    mode: str = "auto"
    margin: float = 2.5
    floor_dbfs: float = -42.0
    tail_ms: float = 250
    start_frames: int = 4
    init_coupling: float = 0.35
    speaking_threshold: float = 0.8
    calib_ms: float = 300         # at the start of each speech burst everything is echo: learn the leakage

    @classmethod
    def from_dict(cls, d: dict) -> EchoConfig:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class EchoGate:
    def __init__(self, cfg: EchoConfig | None = None):
        self.cfg = cfg or EchoConfig()
        self.coupling = self.cfg.init_coupling
        self._speaking_until = -1.0   # clock time until which we consider her audible (incl. tail)
        self._active = False
        self._loud_run = 0
        self._hold = 0
        self._now = 0.0
        self._calib_until = -1.0

    def reset(self) -> None:
        self._loud_run = 0

    def note_playback(self, now: float, playing: bool) -> None:
        """Call every frame with whether she is audible right now."""
        self._now = now
        if playing:
            if not self._active:
                self._calib_until = now + self.cfg.calib_ms / 1000.0
            self._speaking_until = now + self.cfg.tail_ms / 1000.0
        self._active = now <= self._speaking_until

    @property
    def active(self) -> bool:
        """True while her voice (or its tail) can be in the mic."""
        return self._active and self.cfg.mode != "off"

    def allow(self, mic_rms: float, ref_rms: float) -> bool:
        """May this mic frame be treated as possible user speech?"""
        c = self.cfg
        if not self.active:
            self._loud_run = 0
            return True
        if c.mode == "mute":
            return False
        if self._now < self._calib_until:
            if ref_rms > 1e-4:
                self.coupling += 0.4 * (min(mic_rms / ref_rms, 1.5) - self.coupling)
            return False
        expected = self.coupling * ref_rms
        loud = dbfs(mic_rms) > c.floor_dbfs and mic_rms > c.margin * expected
        if loud:
            self._loud_run += 1
            # sustained before we believe it: one clipped echo peak is not a user
            if self._loud_run >= c.start_frames:
                self._hold = HOLD_FRAMES
                return True
            return self._hold > 0
        self._loud_run = 0
        if self._hold > 0:
            self._hold -= 1
            return True
        # learn only from frames that look like the echo we already expect, never from the user's onset
        if ref_rms > 1e-4 and mic_rms <= 1.3 * expected + 1e-4:
            self.coupling += 0.2 * (mic_rms / ref_rms - self.coupling)
        return False

    def clean(self, frame: np.ndarray, reference: np.ndarray, rate: int) -> np.ndarray:
        """Subtract correlated playback even when its gain/delay differs from the RMS prediction."""
        ref = resample(reference, rate, MIC_SR)
        if ref.size < frame.size or rms(frame) < 1e-4:
            return frame
        n = frame.size
        cross = np.correlate(ref, frame, mode="valid")
        power = np.cumsum(np.square(ref), dtype=np.float64)
        power = power[n - 1:] - np.r_[0.0, power[:-n]]
        score = cross ** 2 / np.maximum(power, 1e-8)
        i = int(np.argmax(score))
        if score[i] < 0.2 * float(np.dot(frame, frame)):
            return frame
        echo = ref[i:i + n] * (cross[i] / power[i])
        residual = frame - echo
        if rms(residual) < max(10 ** (self.cfg.floor_dbfs / 20), 0.15 * rms(echo)):
            return np.zeros_like(frame)
        return residual

    def vad_threshold(self, base: float) -> float:
        return max(base, self.cfg.speaking_threshold) if self.active and self.cfg.mode == "auto" else base
