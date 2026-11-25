"""Benchmark one TTS backend: time to first audio, real-time factor, VRAM, WAVs for scoring.

    python -m companion.voice.bench.tts_bench --backend kokoro --out out/kokoro
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import tts
from .sentences import EMOTION_FOR, SENTENCES


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = gpu_used_mib()
            if v:
                self.peak = max(self.peak, v)
            time.sleep(0.1)

    def stop(self):
        self._stop.set()


def run(backend_name: str, out: Path, mixed_emotions: bool, options: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    base = gpu_used_mib()
    sampler = VramSampler()
    sampler.start()
    backend = tts.create(backend_name, **options)
    t0 = time.perf_counter()
    backend.load()
    load_s = time.perf_counter() - t0
    after_load = gpu_used_mib()
    # warm-up (kernel compile, caches), not counted
    for _ in range(2):
        for _chunk in backend.stream("Olá, tudo bem com você hoje?", "neutral"):
            pass
    rows = []
    for i, text in enumerate(SENTENCES):
        emotion = EMOTION_FOR[i] if mixed_emotions else "neutral"
        t = time.perf_counter()
        first = None
        chunks = []
        for c in backend.stream(text, emotion):
            if c.size and first is None:
                first = time.perf_counter() - t
            chunks.append(c)
        total = time.perf_counter() - t
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        dur = audio.size / backend.sample_rate
        sf.write(str(out / f"{i:02d}_{emotion}.wav"), audio, backend.sample_rate)
        rows.append({"i": i, "emotion": emotion, "first_audio_s": first, "synth_s": total,
                     "audio_s": dur, "rtf": total / dur if dur else None, "words": len(text.split())})
        print(f"[{backend_name}] {i} {emotion:9s} first={first and round(first, 3)} total={total:.2f}s audio={dur:.2f}s", flush=True)
    time.sleep(0.3)
    sampler.stop()
    firsts = [r["first_audio_s"] for r in rows if r["first_audio_s"] is not None]
    summary = {
        "backend": backend_name, "sample_rate": backend.sample_rate, "load_s": round(load_s, 1),
        "vram_baseline_mib": base, "vram_after_load_mib": after_load, "vram_peak_mib": sampler.peak,
        "vram_delta_peak_mib": (sampler.peak - base) if base else None,
        "first_audio_p50_s": round(statistics.median(firsts), 3) if firsts else None,
        "first_audio_max_s": round(max(firsts), 3) if firsts else None,
        "rtf_mean": round(statistics.mean(r["rtf"] for r in rows if r["rtf"]), 3),
        "rows": rows, "options": options,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    backend.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--neutral", action="store_true", help="all sentences neutral (default: one emotion each)")
    ap.add_argument("--opt", action="append", default=[], help="key=value backend option")
    a = ap.parse_args()
    options = {}
    for kv in a.opt:
        k, v = kv.split("=", 1)
        options[k] = int(v) if v.isdigit() else v
    s = run(a.backend, a.out, not a.neutral, options)
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}))


if __name__ == "__main__":
    main()
