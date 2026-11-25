"""Echo gate: her own voice in the mic must not count as the user; the user over her voice must."""

import asyncio

import numpy as np
import pytest
from conftest import silence, speech

from companion.voice import Voice
from companion.voice.audio import FRAME, MIC_SR, rms
from companion.voice.echo import EchoConfig, EchoGate
from companion.voice.mic import ArrayMic
from companion.voice.sink import VirtualSink
from companion.voice.tts.fake import FakeTts
from companion.voice.vad import EnergyVad

FRAME_S = FRAME / MIC_SR


def gate(**kw):
    kw.setdefault("calib_ms", 0)   # tests of the steady-state logic skip the learning window
    return EchoGate(EchoConfig(**kw))


def run_frames(g, mic_levels, ref, t0=0.0, playing=True):
    out = []
    for i, m in enumerate(mic_levels):
        g.note_playback(t0 + i * FRAME_S, playing)
        out.append(g.allow(m, ref))
    return out


def test_gate_is_open_when_she_is_silent():
    g = gate()
    assert all(run_frames(g, [0.001, 0.05, 0.3], ref=0.0, playing=False))
    assert not g.active


def test_echo_level_input_is_blocked_while_she_speaks():
    g = gate(init_coupling=0.35)
    assert not any(run_frames(g, [0.35 * 0.2] * 40, ref=0.2))


def test_user_clearly_louder_than_the_echo_is_allowed_after_a_few_frames():
    g = gate(start_frames=4, margin=2.5)
    res = run_frames(g, [0.30] * 10, ref=0.1)   # expected echo 0.035, user 0.30
    assert res[:3] == [False, False, False]
    assert all(res[3:])


def test_a_single_loud_frame_is_not_a_user():
    g = gate(start_frames=4)
    res = run_frames(g, [0.01, 0.01, 0.4, 0.01, 0.01, 0.4, 0.01], ref=0.2)
    assert not any(res)


def test_quiet_mic_is_never_the_user_even_if_she_is_silent_in_the_reference():
    g = gate(floor_dbfs=-42.0)
    assert not any(run_frames(g, [0.002] * 20, ref=0.0005))   # -54 dBFS


def test_leakage_is_learned_at_the_start_of_her_speech_so_a_loud_room_stays_quiet():
    g = gate(init_coupling=0.05, margin=2.5, start_frames=2, calib_ms=300)
    res = run_frames(g, [0.6 * 0.2] * 60, ref=0.2)      # real leakage 0.6, far above the initial guess
    assert not any(res)
    assert g.coupling == pytest.approx(0.6, abs=0.1)


def test_a_naive_initial_guess_alone_would_have_taken_that_for_the_user():
    g = gate(init_coupling=0.05, margin=2.5, start_frames=2, calib_ms=0)
    assert any(run_frames(g, [0.6 * 0.2] * 10, ref=0.2))


def test_gate_stays_closed_for_the_tail_after_playback_ends():
    g = gate(tail_ms=250)
    g.note_playback(0.0, True)
    g.note_playback(0.2, False)
    assert g.active
    g.note_playback(0.5, False)
    assert not g.active


def test_mute_mode_is_deaf_while_she_speaks():
    g = gate(mode="mute")
    assert not any(run_frames(g, [0.9] * 10, ref=0.1))


def test_off_mode_never_gates():
    g = gate(mode="off")
    assert all(run_frames(g, [0.001] * 5, ref=0.5))


def test_vad_threshold_is_stricter_while_speaking_only():
    g = gate(speaking_threshold=0.8)
    assert g.vad_threshold(0.5) == 0.5
    g.note_playback(0.0, True)
    assert g.vad_threshold(0.5) == 0.8


# ---- the whole path: mic = her voice (leaking) + optionally the user ----------------------------

HER = speech(2.0, amp=0.5, seed=0)


def scenario(user_at=None, mode="auto", coupling=0.3):
    """Real-time run. She speaks from t=0.3 s; the mic hears her at `coupling`, plus the user from `user_at`."""
    tts = FakeTts(sample_rate=MIC_SR)
    tts.stream = lambda text, emotion, cancel=None: iter([HER[i:i + 1600] for i in range(0, HER.size, 1600)])
    sink = VirtualSink(MIC_SR)
    mic_audio = np.concatenate([silence(0.3), coupling * HER, silence(1.5)])
    if user_at is not None:
        user = speech(1.2, amp=0.9, seed=2)
        i = int(user_at * MIC_SR)
        mic_audio[i:i + user.size] += user[: mic_audio.size - i]
    mic = ArrayMic(mic_audio, realtime=True, tail_s=0.3)
    starts, utts = [], []

    async def go():
        cfg = {"echo": {"mode": mode}, "listen": {"vad": {"min_speech_ms": 200, "end_silence_ms": 400}},
               "stt": {"backend": "fake", "text": "oi"}}
        v = Voice(cfg, on_speech_start=lambda: starts.append(asyncio.get_running_loop().time()),
                  mic=mic, tts=tts, sink=sink, vad=EnergyVad(margin_db=14))
        got = []

        async def reader():
            async for u in v.listen():
                got.append(u)

        r = asyncio.create_task(reader())
        await asyncio.sleep(0.3)
        t_speak = asyncio.get_running_loop().time()
        sp = asyncio.create_task(v.speak("frase longa de teste", "neutral", lambda w: None))
        await asyncio.wait_for(asyncio.gather(sp, r), timeout=6)
        await v.close()
        return got, t_speak, sink

    got, t_speak, sink = asyncio.run(go())
    return starts, got, t_speak, sink


def test_her_own_voice_in_the_mic_does_not_trigger_barge_in():
    starts, got, _t, sink = scenario(user_at=None)
    assert starts == []
    assert got == []
    assert sink.written_s == pytest.approx(2.0, abs=0.05)   # she finished her whole sentence


def test_without_the_gate_her_echo_would_trigger_it():
    # control: proves the scenario really is hostile to a naive VAD
    starts, got, _t, _s = scenario(user_at=None, mode="off")
    assert starts, "echo was not loud enough to matter; the test above proves nothing"


def test_user_talking_over_her_triggers_barge_in_and_stops_playback_fast():
    starts, got, t_speak, sink = scenario(user_at=1.0)
    assert len(starts) == 1
    assert 1.0 - 0.3 <= starts[0] - t_speak + 0.3 <= 1.0 + 0.6      # fires within ~0.6 s of the user starting
    assert sink.buffered_s == 0.0
    assert sink.written_s < 1.3                                      # she was cut off, not finished
    assert len(got) == 1 and got[0].text == "oi"
