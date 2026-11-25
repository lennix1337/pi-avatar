"""Peak GPU memory with TTS and STT loaded together and used once each (the real working set).

    python -m companion.voice.bench.vram_total --tts kokoro --stt faster-whisper
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..audio import MIC_SR
from ..stt import create_stt
from ..tts import create as create_tts
from .sentences import SENTENCES
from .tts_bench import VramSampler, gpu_used_mib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", required=True)
    ap.add_argument("--stt", default="faster-whisper")
    a = ap.parse_args()
    base = gpu_used_mib() or 0
    sampler = VramSampler()
    sampler.start()
    tts = create_tts(a.tts)
    tts.load()
    stt = create_stt({"backend": a.stt})
    stt.load()
    after_load = gpu_used_mib() or 0
    audio = np.concatenate([tts.synth(s, "happy") for s in SENTENCES[:3]])
    ratio = tts.sample_rate // MIC_SR if tts.sample_rate % MIC_SR == 0 else None
    pcm = audio[::ratio] if ratio else np.interp(np.linspace(0, audio.size, int(audio.size * MIC_SR / tts.sample_rate)), np.arange(audio.size), audio)
    text = stt.transcribe(pcm.astype(np.float32))
    time.sleep(0.5)
    sampler.stop()
    out = {"tts": a.tts, "stt": a.stt, "baseline_mib": base, "after_load_mib": after_load,
           "peak_mib": sampler.peak, "delta_after_load_mib": after_load - base, "delta_peak_mib": sampler.peak - base,
           "transcript": text}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
