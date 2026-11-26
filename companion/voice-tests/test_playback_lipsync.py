"""Drive the hardware sink with a deterministic PortAudio clock (no sound card)."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from companion.voice.lipsync import StreamingLipSync
from companion.voice.sink import SoundDeviceSink
from conftest import VOWELS, synth_vowel


@pytest.fixture
def output(monkeypatch):
    stream = SimpleNamespace(latency=0.02, time=0.0)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **_: stream))
    return SoundDeviceSink(16000), stream


def test_mouth_edges_follow_the_pcm_at_the_dac_within_50_ms(output, record_property):
    sink, stream = output
    sr = sink.sample_rate
    block = sr // 50
    tone = synth_vowel(VOWELS["aa"], dur=0.2)
    pcm = np.concatenate((np.zeros(4 * block, np.float32), tone, np.zeros(10 * block, np.float32)))
    lip = StreamingLipSync(sr)
    lip.feed(pcm, final=True)
    sink.write(pcm)

    # The per-buffer timestamp is 140 ms ahead of the callback, not the stream's nominal 20 ms.
    first_dac = 0.14
    rendered = []
    callback = 0
    for tick in range(25):
        stream.time = tick / 30
        while callback * 0.02 <= stream.time:
            out = np.empty((block, 1), np.float32)
            dac = first_dac + callback * 0.02
            sink._callback(out, block, SimpleNamespace(outputBufferDacTime=dac), None)
            start = callback * block
            np.testing.assert_array_equal(out[:, 0], pcm[start:start + block] if start < len(pcm) else 0)
            callback += 1
        pos, audible = sink.playback()
        rendered.append((stream.time, audible and sum(lip.frame_at(pos).values()) > 0.2))

    opened = next(t for t, mouth_open in rendered if mouth_open)
    closed = next(t for t, mouth_open in rendered if t > opened and not mouth_open)
    expected = (first_dac + 0.08, first_dac + 0.28)
    record_property("max_edge_error_ms", round(1000 * max(abs(t - want) for t, want in zip((opened, closed), expected)), 2))
    assert (opened, closed) == pytest.approx(expected, abs=0.05)
    assert sink.playback() == (len(pcm) / sr, False)


def test_dac_padding_does_not_advance_the_viseme_timeline(output):
    sink, stream = output
    out = np.empty((320, 1), np.float32)
    for dac in (0.1, 0.2):
        # 10 ms of PCM followed by 10 ms of device padding. The callbacks can be queued ahead.
        sink.write(np.ones(160, np.float32))
        sink._callback(out, len(out), SimpleNamespace(outputBufferDacTime=dac), None)
        np.testing.assert_array_equal(out[160:], 0.0)
    for wall, pos, audible in [(0.09, 0.0, False), (0.105, 0.005, True), (0.115, 0.01, False),
                                (0.19, 0.01, False), (0.205, 0.015, True), (0.22, 0.02, False)]:
        stream.time = wall
        actual, playing = sink.playback()
        assert actual == pytest.approx(pos)
        assert playing == audible


async def test_barge_in_aborts_dac_queue_without_blocking_the_loop(monkeypatch, record_property):
    import asyncio
    import threading
    import time

    loop = asyncio.get_running_loop()
    aborting = asyncio.Event()
    release = threading.Event()
    stopped = []
    stream = SimpleNamespace(latency=0.3, time=0.0, active=True)

    def abort():
        loop.call_soon_threadsafe(aborting.set)
        assert release.wait(1), "PortAudio abort blocked the event loop"
        stream.active = False
        stopped.append(time.perf_counter())

    stream.abort = abort
    stream.start = lambda: setattr(stream, "active", True)
    stream.stop = stream.close = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **_: stream))
    sink = SoundDeviceSink(16000)
    sink.write(np.full(3200, 0.25, np.float32))
    sink.take(3200, dac_start=0.01)
    stream.time = 0.02
    assert sink.playback()[1]
    # This is the path Speaker.cancel uses; no software PCM remains, but DAC still has 190 ms.
    t0 = time.perf_counter()
    sink.clear()
    try:
        await asyncio.wait_for(aborting.wait(), 0.1)
        assert not sink.playback()[1]
    finally:
        release.set()
    await sink.prepare()
    stop_ms = (stopped[0] - t0) * 1000
    record_property("cancel_to_dac_abort_ms", round(stop_ms, 2))
    assert stop_ms < 100
    assert stream.active
    # New playback must not be displaced or preceded by the discarded schedule.
    sink.write(np.full(320, 0.1, np.float32))
    assert sink.take(320, dac_start=0.03) == pytest.approx(np.full(320, 0.1))
    stream.time = 0.035
    assert sink.playback() == (pytest.approx(0.205), True)
    await sink.shutdown()
