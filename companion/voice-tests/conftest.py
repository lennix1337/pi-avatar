import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _resonator(x, f, bw, sr):
    from scipy.signal import lfilter

    r = np.exp(-np.pi * bw / sr)
    th = 2 * np.pi * f / sr
    return lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)


def synth_vowel(formants, dur=0.5, sr=16000, f0=220.0, amp=0.3, seed=0):
    """Impulse-train glottal source through formant resonators."""
    n = int(dur * sr)
    src = np.zeros(n)
    src[:: int(sr / f0)] = 1.0
    y = src
    for f in formants:
        y = _resonator(y, f, 90.0, sr)
    y = y / (np.abs(y).max() + 1e-9) * amp
    return y.astype(np.float32)


VOWELS = {
    "aa": (850, 1250, 2800),
    "ih": (320, 2750, 3300),
    "ou": (360, 900, 2600),
    "ee": (620, 2250, 3000),
    "oh": (520, 930, 2700),
}


@pytest.fixture
def vowels():
    return {k: synth_vowel(v) for k, v in VOWELS.items()}


def tone_burst(dur, sr=16000, f=220.0, amp=0.3):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) / 1.5).astype(np.float32)
