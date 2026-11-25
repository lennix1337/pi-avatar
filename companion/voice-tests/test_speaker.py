"""Sentence pipelining, playback-clock lip sync and barge-in cancel."""

import asyncio
import time

import numpy as np
import pytest

from companion.voice.lipsync import VISEMES
from companion.voice.sink import VirtualSink
from companion.voice.speaker import Speaker
from companion.voice.tts.fake import FakeTts

SR = 16000


def make(delay=0.0, secs_per_word=0.2, **kw):
    tts = FakeTts(sample_rate=SR, delay=delay, secs_per_word=secs_per_word, **kw)
    sink = VirtualSink(SR)
    return tts, sink, Speaker(tts, sink)


def test_next_sentence_is_synthesized_while_current_plays_no_gap():
    # Each sentence takes 0.5 s to start producing audio and then plays for 0.6 s.
    tts, sink, sp = make(delay=0.5, secs_per_word=0.2)

    async def go():
        t = time.monotonic()
        await asyncio.gather(*(sp.speak(f"um dois {i}", "neutral", lambda w: None) for i in range(3)))
        elapsed = time.monotonic() - t
        await sp.close()
        return elapsed

    elapsed = asyncio.run(go())
    serial = 3 * (0.5 + 0.6)
    pipelined = 0.5 + 3 * 0.6
    assert elapsed < pipelined + 0.35, f"{elapsed:.2f}s (serial would be {serial:.2f}s)"
    assert sink.starved_s < 0.06, f"audible gap of {sink.starved_s:.3f}s between sentences"


def test_sentences_play_in_order_and_contiguously_on_the_timeline():
    tts, sink, sp = make(delay=0.05)

    async def go():
        texts = ["a b", "c d e", "f"]
        await asyncio.gather(*(sp.speak(t, "happy", lambda w: None) for t in texts))
        hist = list(sp.history)
        await sp.close()
        return texts, hist

    texts, hist = asyncio.run(go())
    assert [c[0] for c in tts.calls] == texts
    assert [h.text for h in hist] == texts
    for a, b in zip(hist, hist[1:]):
        assert b.start == pytest.approx(a.end, abs=1e-6)


def test_speak_returns_only_after_playback_of_that_sentence():
    tts, sink, sp = make(secs_per_word=0.25)

    async def go():
        t = time.monotonic()
        await sp.speak("um dois três quatro", "neutral", lambda w: None)   # 1.0 s of audio
        elapsed = time.monotonic() - t
        await sp.close()
        return elapsed

    assert 0.95 <= asyncio.run(go()) < 1.3


def test_mouth_frames_arrive_at_30hz_and_end_closed():
    tts, sink, sp = make(secs_per_word=0.25)
    got = []

    async def go():
        await sp.speak("um dois três quatro", "neutral", lambda w: got.append((time.monotonic(), w)))
        await sp.close()

    asyncio.run(go())
    assert all(set(w) == set(VISEMES) and all(0.0 <= v <= 1.0 for v in w.values()) for _, w in got)
    body = got[:-1]
    rate = (len(body) - 1) / (body[-1][0] - body[0][0])
    assert 27 <= rate <= 33, rate
    assert all(v == 0.0 for v in got[-1][1].values())   # the pet gets a closed mouth at the end


def test_mouth_follows_the_audio_not_the_synthesis_speed():
    # Audio is produced instantly (fast TTS) but must be lip-synced over its playing time.
    tts, sink, sp = make(secs_per_word=0.5)
    stamps = []

    async def go():
        await sp.speak("um dois", "neutral", lambda w: stamps.append(time.monotonic()))
        await sp.close()

    asyncio.run(go())
    assert stamps[-2] - stamps[0] == pytest.approx(1.0, abs=0.15)


def test_cancel_silences_within_100ms_and_drops_the_queue():
    tts, sink, sp = make(delay=0.3, secs_per_word=0.5)
    mouths = []

    async def go():
        tasks = [asyncio.create_task(sp.speak(f"um dois {i}", "neutral", mouths.append)) for i in range(4)]
        await asyncio.sleep(0.4)
        before = sink.position()
        t = time.perf_counter()
        sp.cancel()
        took = time.perf_counter() - t
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=0.1)   # speak() calls return at once
        returned = time.perf_counter() - t
        await asyncio.sleep(0.2)
        after = sink.position()
        buffered = sink.buffered_s
        await sp.close()
        return took, returned, before, after, buffered

    took, returned, before, after, buffered = asyncio.run(go())
    assert took < 0.1 and returned < 0.1
    assert buffered == 0.0
    assert after - before < 0.02                    # the playback clock stopped
    assert len(tts.calls) <= 2                      # queued sentences were never synthesized
    assert all(v == 0.0 for v in mouths[-1].values())


def test_cancel_during_synthesis_then_speak_again_works():
    tts, sink, sp = make(delay=0.4, secs_per_word=0.1)

    async def go():
        t1 = asyncio.create_task(sp.speak("primeira frase", "neutral", lambda w: None))
        await asyncio.sleep(0.1)
        sp.cancel()
        await t1
        assert sink.buffered_s == 0.0
        await asyncio.sleep(0.5)                    # the aborted synthesis finishes and must stay silent
        assert sink.buffered_s == 0.0 and sink.written_s == 0.0
        await asyncio.wait_for(sp.speak("segunda", "neutral", lambda w: None), timeout=2)
        await sp.close()

    asyncio.run(go())
    assert tts.calls[-1][0] == "segunda"


def test_tts_failure_reaches_the_caller_and_the_queue_survives():
    tts, sink, sp = make()
    real = tts.stream

    def boom(text, emotion, cancel=None):
        if "ruim" in text:
            raise RuntimeError("model exploded")
        yield from real(text, emotion, cancel)

    tts.stream = boom

    async def go():
        with pytest.raises(RuntimeError, match="exploded"):
            await sp.speak("frase ruim", "neutral", lambda w: None)
        await asyncio.wait_for(sp.speak("frase boa", "neutral", lambda w: None), timeout=2)
        await sp.close()

    asyncio.run(go())


def test_a_failing_mouth_callback_does_not_stop_playback():
    tts, sink, sp = make()

    def bad(_):
        raise ValueError("pet disconnected")

    async def go():
        await asyncio.wait_for(sp.speak("um dois", "neutral", bad), timeout=2)
        await sp.close()

    asyncio.run(go())
    assert sink.written_s > 0


def test_resamples_to_the_sink_rate():
    tts = FakeTts(sample_rate=24000)
    sink = VirtualSink(16000)
    sp = Speaker(tts, sink)

    async def go():
        await sp.speak("um dois", "neutral", lambda w: None)
        await sp.close()

    asyncio.run(go())
    assert sink.written_s == pytest.approx(0.4, abs=0.01)
