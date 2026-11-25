"""Voice facade: WAV file in -> Utterance out, push-to-talk, and TTS -> WAV -> STT end to end."""

import asyncio
import os
import time

import numpy as np
import pytest
import soundfile as sf
from conftest import noise, silence, speech

from companion.voice import Utterance, Voice
from companion.voice.audio import MIC_SR
from companion.voice.tts.fake import FakeTts
from companion.voice.vad import EnergyVad

CFG = {"stt": {"backend": "fake", "text": ["primeira fala", "segunda fala"]},
       "listen": {"vad": {"backend": "energy"}}}


def collect(v, n, timeout=8):
    async def go():
        out = []

        async def rd():
            async for u in v.listen():
                out.append(u)
                if len(out) == n:
                    return

        await asyncio.wait_for(rd(), timeout)
        await v.close()
        return out

    return asyncio.run(go())


def test_wav_file_is_segmented_and_transcribed(tmp_path):
    audio = np.concatenate([noise(0.8), speech(1.2), noise(1.0), speech(1.5, seed=2), noise(1.0)])
    p = tmp_path / "talk.wav"
    sf.write(str(p), audio, MIC_SR)
    cfg = {**CFG, "listen": {**CFG["listen"], "source": f"file:{p}"}}
    starts = []
    v = Voice(cfg, on_speech_start=lambda: starts.append(1), vad=EnergyVad())
    utts = collect(v, 2)
    assert [u.text for u in utts] == ["primeira fala", "segunda fala"]
    assert all(isinstance(u, Utterance) for u in utts)
    assert utts[0].ended_at > utts[0].started_at
    assert utts[1].started_at > utts[0].ended_at
    assert utts[0].started_at == pytest.approx(utts[0].started_at, abs=1)
    assert utts[1].started_at - utts[0].started_at == pytest.approx(2.2 + 0.0, abs=0.4)
    assert len(starts) == 2


def test_empty_transcripts_are_not_reported(tmp_path):
    audio = np.concatenate([noise(0.5), speech(1.0), noise(1.0), speech(1.0, seed=1), noise(1.0)])
    p = tmp_path / "t.wav"
    sf.write(str(p), audio, MIC_SR)
    cfg = {"stt": {"backend": "fake", "text": ["", "olá"]},
           "listen": {"vad": {"backend": "energy"}, "source": f"file:{p}"}}
    utts = collect(Voice(cfg, vad=EnergyVad()), 1)
    assert [u.text for u in utts] == ["olá"]


def test_push_to_talk_records_only_while_the_key_is_down():
    from companion.voice.mic import ArrayMic

    mic = ArrayMic(np.concatenate([speech(3.0)]), realtime=True, tail_s=0.5)
    cfg = {"stt": {"backend": "fake", "text": "oi"}, "listen": {"mode": "ptt", "vad": {"backend": "energy"}}}
    v = Voice(cfg, mic=mic)

    async def go():
        out = []

        async def rd():
            async for u in v.listen():
                out.append(u)
                return

        r = asyncio.create_task(rd())
        await asyncio.sleep(0.5)
        v.ptt(True)
        await asyncio.sleep(1.0)
        v.ptt(False)
        await asyncio.wait_for(r, 4)
        await v.close()
        return out

    utts = asyncio.run(go())
    assert utts[0].text == "oi"
    assert utts[0].ended_at - utts[0].started_at == pytest.approx(1.0, abs=0.2)
    assert v._stt.calls[0] / MIC_SR == pytest.approx(1.0, abs=0.2)   # STT got ~1 s, not the whole file


def test_speak_and_cancel_through_the_facade():
    tts = FakeTts(sample_rate=16000, secs_per_word=0.5)
    v = Voice({"output": {"backend": "virtual"}}, tts=tts)
    frames = []

    async def go():
        t = asyncio.create_task(v.speak("um dois três quatro", "happy", frames.append))
        await asyncio.sleep(0.5)
        assert v.speaking
        t0 = time.perf_counter()
        v.cancel()
        dt = time.perf_counter() - t0
        await asyncio.wait_for(t, 0.2)
        assert not v.speaking
        await v.close()
        return dt

    assert asyncio.run(go()) < 0.1
    assert frames and all(x == 0.0 for x in frames[-1].values())
    assert tts.calls == [("um dois três quatro", "happy")]


def test_barge_in_cancels_speech_and_notifies_the_brain(tmp_path):
    from companion.voice.mic import ArrayMic

    her = speech(3.0, amp=0.5)
    tts = FakeTts(sample_rate=16000)
    tts.stream = lambda text, emotion, cancel=None: iter([her[i:i + 1600] for i in range(0, her.size, 1600)])
    mic = ArrayMic(np.concatenate([silence(0.5), speech(1.5, amp=0.9, seed=2), silence(1.0)]), realtime=True, tail_s=0.2)
    called = []
    v = Voice({"output": {"backend": "virtual"}, "stt": {"backend": "fake", "text": "pare"}},
              on_speech_start=lambda: called.append(time.monotonic()), mic=mic, tts=tts, vad=EnergyVad())

    async def go():
        out = []

        async def rd():
            async for u in v.listen():
                out.append(u)
                return

        r = asyncio.create_task(rd())
        t_speak = time.monotonic()
        speaking = asyncio.create_task(v.speak("frase", "neutral", lambda w: None))
        await asyncio.wait_for(asyncio.gather(speaking, r), 6)
        await v.close()
        return out, t_speak

    out, t_speak = asyncio.run(go())
    assert len(called) == 1 and called[0] - t_speak < 1.4
    assert v._sink.written_s < 2.0 and out[0].text == "pare"


@pytest.mark.skipif(not os.environ.get("COMPANION_VOICE_E2E"), reason="needs models: set COMPANION_VOICE_E2E=1")
def test_tts_sentence_to_wav_to_listen_returns_the_same_words(tmp_path):
    """Real TTS renders a sentence to a WAV; the real listening path reads it back as a microphone."""
    from companion.voice import config as cfgmod
    from companion.voice.audio import write_wav
    from companion.voice.tts import create as create_tts

    text = "Oi, tudo bem? Hoje eu acordei com vontade de conversar sobre o universo."
    user = {}
    if os.environ.get("COMPANION_VOICE_CFG"):
        import json

        user = json.loads(os.environ["COMPANION_VOICE_CFG"])
    cfg = cfgmod.merge(user)
    tts = create_tts(cfg["tts"]["backend"], **cfg["tts"].get("options", {}))
    tts.load()
    audio = np.concatenate([silence(0.8), tts.synth(text, "neutral"), silence(1.2)]).astype(np.float32)
    wav = tmp_path / "sentence.wav"
    write_wav(wav, audio, tts.sample_rate)
    tts.close()
    cfg["listen"]["source"] = f"file:{wav}"
    v = Voice(cfg)
    utts = collect(v, 1, timeout=120)

    def words(s):
        import re
        return re.findall(r"\w+", s.lower())

    ref, hyp = words(text), words(utts[0].text)
    hits = sum(1 for w in ref if w in hyp)
    assert hits / len(ref) >= 0.8, (utts[0].text, text)


def test_canary_files_are_discovered_in_the_standard_places(tmp_path):
    from companion.voice.stt import find_canary_model, find_node_modules

    assert find_canary_model(tmp_path) == "" and find_node_modules(tmp_path) == ""
    model = tmp_path / ".cache/huggingface/hub/models--handy-computer--canary-1b-v2-gguf/snapshots/abc/canary-1b-v2-Q8_0.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    pkg = tmp_path / ".pi/agent/npm/node_modules/transcribe-cpp"
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text("{}")
    assert find_canary_model(tmp_path) == str(model)
    assert find_node_modules(tmp_path) == str(pkg.parent)
