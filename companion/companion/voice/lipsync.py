"""Audio -> viseme weights (aa, ih, ou, ee, oh) for the pet's mouth.

Per frame (default 30 Hz): loudness gives how open the mouth is, a voicing check
keeps fricatives and noise from flapping the jaw, and the first two formants
(LPC) pick the vowel shape. The result is smoothed with a fast attack and a
slower release so the mouth moves like a mouth, not like a VU meter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

VISEMES = ("aa", "ih", "ou", "ee", "oh")

# Typical (F1, F2) in Hz of a female voice for the vowels each VRM viseme shows.
VOWEL_FORMANTS: dict[str, tuple[float, float]] = {
    "aa": (850.0, 1250.0),  # a
    "ih": (360.0, 2750.0),  # i
    "ou": (430.0, 900.0),   # u (LPC reads low F1 a bit high)
    "ee": (620.0, 2250.0),  # e
    "oh": (550.0, 930.0),   # o
}

ANALYSIS_SR = 16000


def zero_mouth() -> dict[str, float]:
    return {v: 0.0 for v in VISEMES}


def _bark(f: float) -> float:
    return 26.81 * f / (1960.0 + f) - 0.53


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def _levinson(r: np.ndarray, order: int) -> np.ndarray:
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / err
        prev = a.copy()
        for j in range(1, i):
            a[j] = prev[j] + k * prev[i - j]
        a[i] = k
        err *= 1.0 - k * k
        if err <= 1e-12:
            break
    return a


def formants(frame: np.ndarray, sr: int = ANALYSIS_SR, order: int | None = None) -> list[float]:
    """First formants (Hz, ascending) of a voiced frame by LPC root solving."""
    order = order or (2 + sr // 1000)
    x = np.asarray(frame, dtype=np.float64)
    if x.size < order * 2:
        return []
    x = x - x.mean()
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]) * np.hanning(x.size)
    n = x.size
    r = np.array([np.dot(x[: n - k], x[k:]) for k in range(order + 1)])
    if r[0] <= 1e-9:
        return []
    r[0] *= 1.0001  # white-noise correction keeps the filter stable
    a = _levinson(r, order)
    out = []
    for z in np.roots(a):
        if z.imag <= 0.01:
            continue
        f = math.atan2(z.imag, z.real) * sr / (2 * math.pi)
        bw = -math.log(max(abs(z), 1e-6)) * sr / math.pi
        if 200.0 < f < sr / 2 - 200 and bw < 500.0:
            out.append(f)
    return sorted(out)


def vowel_probs(f1: float, f2: float, temperature: float = 0.9) -> dict[str, float]:
    """Soft vowel membership from distance to templates in Bark space."""
    b1, b2 = _bark(f1), _bark(f2)
    w = {}
    for v, (t1, t2) in VOWEL_FORMANTS.items():
        d = math.hypot(b1 - _bark(t1), 0.8 * (b2 - _bark(t2)))
        w[v] = math.exp(-((d / temperature) ** 2))
    s = sum(w.values()) or 1.0
    return {v: x / s for v, x in w.items()}


@dataclass
class LipSyncConfig:
    fps: int = 30
    window_ms: float = 40.0
    floor_db: float = -55.0       # below this the mouth is shut
    attack: float = 0.65          # smoothing coefficient when opening (0..1, higher = faster)
    release: float = 0.65         # close promptly at syllable boundaries; the pet also eases the pose
    gain: float = 1.0
    max_total: float = 1.15       # cap on the sum of weights


@dataclass
class VisemeAnalyzer:
    """Stateful: feed it audio, get one weight dict per `1/fps` of audio."""

    sr: int
    cfg: LipSyncConfig = field(default_factory=LipSyncConfig)

    def __post_init__(self) -> None:
        self._state = zero_mouth()
        self._ref_rms = 0.02  # slow estimate of the loud-vowel level, adapts per voice

    def reset(self) -> None:
        self._state = zero_mouth()

    def _raw(self, frame16: np.ndarray) -> dict[str, float]:
        cfg = self.cfg
        rms = float(np.sqrt(np.mean(frame16**2) + 1e-12))
        db = 20.0 * math.log10(rms + 1e-9)
        if db < cfg.floor_db:
            return zero_mouth()
        # adapt the reference level: quick up, very slow down
        if rms > self._ref_rms:
            self._ref_rms += 0.25 * (rms - self._ref_rms)
        else:
            self._ref_rms += 0.002 * (rms - self._ref_rms)
        openness = min(1.0, (rms / max(self._ref_rms, 1e-4)) ** 0.8)
        openness *= min(1.0, (db - cfg.floor_db) / 12.0)  # fade near the floor

        # voicing: few zero crossings and most of the energy under 3.5 kHz
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame16).astype(np.int8)))))
        spec = np.abs(np.fft.rfft(frame16 * np.hanning(frame16.size))) ** 2
        freqs = np.fft.rfftfreq(frame16.size, 1.0 / ANALYSIS_SR)
        low_ratio = spec[freqs < 3500].sum() / (spec.sum() + 1e-12)
        voiced = float(np.clip((0.45 - zcr) / 0.25, 0.0, 1.0) * np.clip((low_ratio - 0.6) / 0.3, 0.0, 1.0))

        out = zero_mouth()
        fm = formants(frame16, ANALYSIS_SR)
        if voiced > 0.05 and len(fm) >= 2:
            f1 = fm[0]
            f2 = next((f for f in fm[1:] if f > f1 + 250), fm[1])
            sharp = {v: p**1.5 for v, p in vowel_probs(f1, f2).items()}
            s = sum(sharp.values()) or 1.0
            for v in VISEMES:
                out[v] = openness * voiced * sharp[v] / s
        # unvoiced hiss (s, f, x): small, narrow smile
        hiss = openness * (1.0 - voiced) * 0.25
        out["ih"] += hiss
        out["ee"] += hiss * 0.5
        total = sum(out.values())
        if total > cfg.max_total:
            out = {k: v * cfg.max_total / total for k, v in out.items()}
        return {k: min(1.0, v * cfg.gain) for k, v in out.items()}

    def _smooth(self, raw: dict[str, float]) -> dict[str, float]:
        cfg = self.cfg
        for v in VISEMES:
            a = cfg.attack if raw[v] > self._state[v] else cfg.release
            self._state[v] += a * (raw[v] - self._state[v])
        return {v: round(float(self._state[v]), 3) for v in VISEMES}

    def track(self, audio: np.ndarray) -> list[dict[str, float]]:
        """One weight dict per 1/fps seconds, frame k centred on k/fps."""
        cfg = self.cfg
        x = resample(np.asarray(audio, dtype=np.float32), self.sr, ANALYSIS_SR)
        hop = ANALYSIS_SR / cfg.fps
        win = int(ANALYSIS_SR * cfg.window_ms / 1000)
        n_frames = max(1, int(math.ceil(x.size / hop)))
        padded = np.concatenate([np.zeros(win // 2, np.float32), x, np.zeros(win, np.float32)])
        return [self._smooth(self._raw(padded[int(k * hop):int(k * hop) + win])) for k in range(n_frames)]


def viseme_track(audio: np.ndarray, sr: int, cfg: LipSyncConfig | None = None) -> list[dict[str, float]]:
    return VisemeAnalyzer(sr, cfg or LipSyncConfig()).track(audio)


class StreamingLipSync:
    """Same frames as `viseme_track`, computed as audio arrives in chunks (any size)."""

    def __init__(self, sr: int, cfg: LipSyncConfig | None = None):
        self.cfg = cfg or LipSyncConfig()
        self.sr = sr
        self._an = VisemeAnalyzer(ANALYSIS_SR, self.cfg)
        self.frames: list[dict[str, float]] = []
        self._win = int(ANALYSIS_SR * self.cfg.window_ms / 1000)
        self._hop = ANALYSIS_SR / self.cfg.fps
        self._x = np.zeros(self._win // 2, np.float32)   # leading pad: frame k is centred on k/fps
        self._n = 0                                      # real 16 kHz samples fed so far
        self._final = False

    @property
    def duration(self) -> float:
        return self._n / ANALYSIS_SR

    def feed(self, chunk: np.ndarray | None = None, final: bool = False) -> None:
        if chunk is not None and len(chunk):
            r = resample(np.asarray(chunk, dtype=np.float32), self.sr, ANALYSIS_SR)
            self._x = np.concatenate([self._x, r])
            self._n += r.size
        if final and not self._final:
            self._final = True
            self._x = np.concatenate([self._x, np.zeros(self._win, np.float32)])
        while True:
            start = int(len(self.frames) * self._hop)
            if start + self._win > self._x.size:
                break
            if self._final and start >= self._win // 2 + self._n:
                break
            self.frames.append(self._an._smooth(self._an._raw(self._x[start:start + self._win])))

    def frame_at(self, t: float) -> dict[str, float]:
        if not self.frames:
            return zero_mouth()
        # Analysis windows are centered, so use the nearest frame, not the previous one.
        frame = int(t * self.cfg.fps + 0.5)
        return self.frames[min(max(0, frame), len(self.frames) - 1)]
