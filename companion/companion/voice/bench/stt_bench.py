"""STT latency (end of speech -> text) and accuracy over the ten benchmark sentences.

The audio is the WAVs a TTS bench wrote, padded like a segmenter turn (0.3 s pre-roll, 0.4 s tail).
The latency is the time `transcribe()` takes; the turn-taking delay before it (end_silence_ms) is added in docs.

    python -m companion.voice.bench.stt_bench --dir out/kokoro --stt faster-whisper
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ..audio import MIC_SR, read_wav_16k
from ..stt import create_stt
from .score import norm, wer
from .sentences import SENTENCES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--stt", default="faster-whisper")
    ap.add_argument("--stt-opt", default="{}")
    a = ap.parse_args()
    stt = create_stt({"backend": a.stt, **json.loads(a.stt_opt)})
    t = time.perf_counter()
    stt.load()
    load_s = time.perf_counter() - t
    files = sorted(a.dir.glob("[0-9][0-9]_*.wav"))
    pad_a, pad_b = np.zeros(int(0.3 * MIC_SR), np.float32), np.zeros(int(0.4 * MIC_SR), np.float32)
    stt.transcribe(np.concatenate([pad_a, read_wav_16k(files[0]), pad_b]))   # warm-up, not counted
    lat, wers = [], []
    for i, f in enumerate(files):
        audio = np.concatenate([pad_a, read_wav_16k(f), pad_b])
        t = time.perf_counter()
        text = stt.transcribe(audio)
        lat.append(time.perf_counter() - t)
        wers.append(wer(norm(SENTENCES[i]), norm(text)))
        print(f"{i} {lat[-1] * 1000:5.0f} ms  wer={wers[-1]:.2f}  {text}")
    lat_s = sorted(lat)
    out = {"stt": a.stt, "load_s": round(load_s, 1), "n": len(lat),
           "p50_ms": round(statistics.median(lat) * 1000), "max_ms": round(lat_s[-1] * 1000),
           "wer_mean": round(float(np.mean(wers)), 4)}
    print(json.dumps(out))
    stt.close()


if __name__ == "__main__":
    main()
