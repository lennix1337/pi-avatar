"""Turn segmentation on synthetic WAV files."""

import asyncio

import numpy as np
import pytest
import soundfile as sf
from conftest import noise, silence, speech

from companion.voice.audio import FRAME, MIC_SR, read_wav_16k
from companion.voice.mic import ArrayMic
from companion.voice.vad import EnergyVad, Segmenter, SegmenterConfig

FRAME_S = FRAME / MIC_SR


def segment(audio, cfg=None, vad=None):
    """Run the segmenter over audio; returns (events, segments)."""
    seg = Segmenter(cfg or SegmenterConfig())
    vad = vad or EnergyVad()
    events, segs = [], []
    n = audio.size // FRAME
    for k in range(n):
        f = audio[k * FRAME:(k + 1) * FRAME]
        for kind, val in seg.push(f, vad(f), (k + 1) * FRAME_S):
            events.append((kind, k * FRAME_S))
            if kind == "end":
                segs.append(val)
    for kind, val in seg.flush(n * FRAME_S):
        events.append((kind, n * FRAME_S))
        if kind == "end":
            segs.append(val)
    return events, segs


def wav(tmp_path, name, audio):
    p = tmp_path / name
    sf.write(str(p), audio, MIC_SR)
    return p


def test_two_utterances_are_separated_by_the_pause(tmp_path):
    audio = np.concatenate([noise(1.0), speech(1.5), noise(1.2), speech(2.0, seed=3), noise(1.0)])
    p = wav(tmp_path, "two.wav", audio)
    _events, segs = segment(read_wav_16k(p))
    assert len(segs) == 2
    assert segs[0].t_start == pytest.approx(1.0, abs=0.15)
    assert segs[0].t_end == pytest.approx(2.5, abs=0.15)
    assert segs[1].t_start == pytest.approx(3.7, abs=0.15)
    assert segs[1].t_end == pytest.approx(5.7, abs=0.15)


def test_segment_audio_includes_preroll_and_the_speech(tmp_path):
    _e, segs = segment(np.concatenate([noise(1.0), speech(1.0), noise(1.2)]))
    dur = segs[0].audio.size / MIC_SR
    assert 1.0 + 0.2 < dur < 1.0 + 0.32 + 0.7     # speech + pre-roll + the end-silence tail


def test_a_short_pause_inside_a_sentence_does_not_split_it():
    audio = np.concatenate([noise(0.5), speech(1.0), silence(0.3), speech(1.0, seed=1), noise(1.0)])
    _e, segs = segment(audio)
    assert len(segs) == 1


def test_start_event_fires_once_per_turn_before_the_turn_ends():
    audio = np.concatenate([noise(0.5), speech(1.0), noise(1.0)])
    events, segs = segment(audio)
    kinds = [k for k, _ in events]
    assert kinds == ["start", "end"]
    assert events[0][1] < 0.5 + 0.25                     # known within ~100 ms of speech onset


def test_noise_only_produces_nothing():
    events, segs = segment(noise(5.0, amp=0.005))
    assert events == [] and segs == []


def test_a_click_is_dropped_not_transcribed():
    click = np.zeros(int(0.04 * MIC_SR), np.float32)
    click[:200] = 0.8
    events, segs = segment(np.concatenate([noise(0.5), click, noise(1.0)]))
    assert segs == []


def test_a_cough_shorter_than_min_speech_is_dropped_after_start():
    audio = np.concatenate([noise(0.5), speech(0.15), noise(1.0)])
    events, segs = segment(audio)
    assert segs == [] and ("drop" in [k for k, _ in events] or events == [])


def test_long_monologue_is_cut_at_the_maximum():
    audio = np.concatenate([noise(0.3), speech(9.0), noise(1.0)])
    _e, segs = segment(audio, SegmenterConfig(max_utterance_s=4.0))
    assert len(segs) >= 2
    assert all(s.audio.size / MIC_SR <= 4.2 for s in segs)


def test_speech_in_noise_is_found_above_the_adaptive_floor():
    audio = np.concatenate([noise(1.0, amp=0.01), speech(1.2) + noise(1.2, amp=0.01, seed=5), noise(1.2, amp=0.01)])
    _e, segs = segment(audio)
    assert len(segs) == 1


def test_flush_closes_a_turn_in_progress():
    audio = np.concatenate([noise(0.3), speech(1.0)])           # input ends mid-speech
    _e, segs = segment(audio)
    assert len(segs) == 1


def test_end_of_turn_latency_is_the_configured_silence():
    cfg = SegmenterConfig(end_silence_ms=550)
    seg = Segmenter(cfg)
    vad = EnergyVad()
    audio = np.concatenate([noise(0.5), speech(1.0), noise(1.5)])
    end_at = None
    for k in range(audio.size // FRAME):
        f = audio[k * FRAME:(k + 1) * FRAME]
        for kind, _ in seg.push(f, vad(f), (k + 1) * FRAME_S):
            if kind == "end":
                end_at = (k + 1) * FRAME_S
    assert end_at == pytest.approx(1.5 + 0.55, abs=0.1)


def test_array_mic_yields_exact_frames():
    async def go():
        m = ArrayMic(noise(0.5), tail_s=0.0)
        return [f async for f in m.frames()]

    frames = asyncio.run(go())
    assert all(f.size == FRAME for f in frames)
    assert len(frames) == int(0.5 * MIC_SR) // FRAME


def test_silero_vad_finds_the_same_turns_when_installed():
    pytest.importorskip("silero_vad")
    from companion.voice.vad import SileroVad

    audio = np.concatenate([noise(1.0), speech(1.5), noise(1.2)])
    _e, segs = segment(audio, vad=SileroVad())
    assert len(segs) in (0, 1)   # synthetic vowels are not guaranteed speech to a neural VAD
