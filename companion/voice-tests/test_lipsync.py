import numpy as np
import pytest

from companion.voice.lipsync import VISEMES, LipSyncConfig, VisemeAnalyzer, formants, viseme_track, zero_mouth
from conftest import synth_vowel, VOWELS


def settled(audio, sr=16000, **cfg):
    tr = viseme_track(audio, sr, LipSyncConfig(**cfg))
    return tr[len(tr) // 2]


@pytest.mark.parametrize("vowel", list(VOWELS))
def test_dominant_viseme_matches_synthetic_vowel(vowel, vowels):
    w = settled(vowels[vowel])
    assert max(w, key=w.get) == vowel, w
    assert w[vowel] > 0.3


def test_formant_estimate_of_a_is_close():
    x = synth_vowel(VOWELS["aa"])[2000:2640]
    f = formants(x)
    assert abs(f[0] - 850) < 150 and any(abs(v - 1250) < 200 for v in f[:3]), f


def test_silence_closes_mouth():
    w = settled(np.zeros(8000, np.float32))
    assert w == zero_mouth()


def test_noise_does_not_open_wide():
    rng = np.random.default_rng(1)
    w = settled((rng.standard_normal(16000) * 0.2).astype(np.float32))
    assert sum(w.values()) < 0.4, w


def test_louder_vowel_opens_more_than_quiet_one():
    sr = 16000
    an = VisemeAnalyzer(sr)
    loud = synth_vowel(VOWELS["aa"], amp=0.5, dur=0.6)
    quiet = synth_vowel(VOWELS["aa"], amp=0.04, dur=0.6)
    seq = np.concatenate([loud, quiet])
    tr = an.track(seq)
    n = len(tr)
    assert tr[n // 4]["aa"] > tr[3 * n // 4]["aa"] + 0.15


def test_frame_rate_is_30hz():
    tr = viseme_track(np.zeros(16000, np.float32), 16000)
    assert len(tr) == 30


def test_other_sample_rates_work(vowels):
    from companion.voice.lipsync import resample

    x24 = resample(vowels["ou"], 16000, 24000)
    w = settled(x24, sr=24000)
    assert max(w, key=w.get) == "ou"


def test_release_is_smoothed_not_instant(vowels):
    audio = np.concatenate([vowels["aa"], np.zeros(8000, np.float32)])
    tr = viseme_track(audio, 16000)
    k = int(0.5 * 30)
    assert tr[k - 2]["aa"] > 0.3
    assert 0.0 <= tr[k + 1]["aa"] < tr[k - 2]["aa"]
    assert all(set(f) == set(VISEMES) and all(0 <= v <= 1 for v in f.values()) for f in tr)
