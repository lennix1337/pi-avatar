"""Try the voice from the command line.

    python -m companion.voice.demo --say "Oi, tudo bem?" --emotion happy --out out.wav
    python -m companion.voice.demo --say "Oi!" --play                 # speakers, with mouth frames printed
    python -m companion.voice.demo --transcribe file.wav
    python -m companion.voice.demo --list-devices                     # enumerate only, opens nothing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from . import config as cfgmod
from .audio import read_wav_16k, write_wav
from .lipsync import viseme_track
from .stt import create_stt
from .tts import create as create_tts


def load_cfg(a: argparse.Namespace) -> dict:
    user = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    cfg = cfgmod.merge(user)
    if a.backend:
        cfg["tts"]["backend"] = a.backend
    if a.stt:
        cfg["stt"]["backend"] = a.stt
    return cfg


def say_to_file(cfg: dict, text: str, emotion: str, out: Path) -> None:
    import numpy as np

    tts = create_tts(cfg["tts"]["backend"], **cfg["tts"].get("options", {}))
    tts.load()
    t0 = time.perf_counter()
    first = None
    chunks = []
    for c in tts.stream(text, emotion):
        if c.size and first is None:
            first = time.perf_counter() - t0
        chunks.append(c)
    total = time.perf_counter() - t0
    audio = np.concatenate(chunks)
    write_wav(out, audio, tts.sample_rate)
    dur = audio.size / tts.sample_rate
    track = viseme_track(audio, tts.sample_rate)
    peak = max(max(f.values()) for f in track)
    print(f"wrote {out}  {dur:.2f}s @ {tts.sample_rate} Hz  first audio {first * 1000:.0f} ms  "
          f"synth {total:.2f}s (RTF {total / dur:.2f})  {len(track)} mouth frames, peak {peak:.2f}")
    tts.close()


async def say_to_speakers(cfg: dict, text: str, emotion: str) -> None:
    from . import Voice

    n = 0

    def on_mouth(w: dict) -> None:
        nonlocal n
        n += 1
        if n % 6 == 0:
            print(" ".join(f"{k}={v:.2f}" for k, v in w.items()))

    v = Voice(cfg)
    await v.speak(text, emotion, on_mouth)
    await v.close()


def transcribe(cfg: dict, path: Path) -> None:
    stt = create_stt(cfg["stt"])
    stt.load()
    audio = read_wav_16k(path)
    stt.transcribe(audio[:16000])     # warm-up, not timed
    t = time.perf_counter()
    text = stt.transcribe(audio)
    dt = time.perf_counter() - t
    print(f"[{stt.name}] {audio.size / 16000:.2f}s of audio in {dt * 1000:.0f} ms")
    print(text)
    stt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--say")
    ap.add_argument("--emotion", default="neutral")
    ap.add_argument("--out", type=Path, help="write the audio here instead of playing it")
    ap.add_argument("--play", action="store_true", help="play through the default output device")
    ap.add_argument("--transcribe", type=Path, metavar="WAV")
    ap.add_argument("--backend", help="TTS backend (default from config)")
    ap.add_argument("--stt", help="STT backend (default from config)")
    ap.add_argument("--config", help="JSON file merged over the defaults")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a)
    if a.list_devices:
        from .sink import list_devices

        for d in list_devices():
            print(d["index"], d["name"], f"in={d['max_input_channels']} out={d['max_output_channels']}")
        return
    if a.say:
        if a.out:
            say_to_file(cfg, a.say, a.emotion, a.out)
        elif a.play:
            asyncio.run(say_to_speakers(cfg, a.say, a.emotion))
        else:
            ap.error("--say needs --out FILE or --play")
    if a.transcribe:
        transcribe(cfg, a.transcribe)
    if not (a.say or a.transcribe):
        ap.error("nothing to do: use --say or --transcribe")


if __name__ == "__main__":
    main()
