"""Defaults for the voice stack; `Voice(config)` deep-merges the caller's dict over these."""

from __future__ import annotations

import copy

DEFAULTS: dict = {
    "tts": {"backend": "voxcpm", "options": {}},
    "stt": {
        "backend": "faster-whisper",   # or "canary" (Pi Voice's transcribe-cpp model through Node)
        "language": "pt",
        "model": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
        "canary": {"model_path": "", "node": "node", "node_modules": ""},
        "min_chars": 2,
    },
    "listen": {
        "mode": "vad",                 # "vad" (hands-free) or "ptt" (push-to-talk via Voice.ptt())
        "input_device": None,          # None = system default (the headset)
        "vad": {
            "backend": "silero",       # falls back to "energy" when silero is not installed
            "threshold": 0.5,
            "neg_threshold": 0.35,
            "start_ms": 96,            # speech this long confirms a turn (fires on_speech_start)
            "min_speech_ms": 250,      # shorter bursts are dropped as noise
            "end_silence_ms": 550,     # silence that ends the turn
            "preroll_ms": 320,
            "max_utterance_s": 30.0,
        },
    },
    "echo": {
        "mode": "auto",                # "auto": energy-vs-playback gate; "mute": deaf while speaking; "off"
        "margin": 2.5,                 # mic must exceed this multiple of the expected echo to count as the user
        "floor_dbfs": -42.0,           # ... and be at least this loud while she speaks
        "tail_ms": 250,                # keep gating this long after playback ends (room tail)
        "start_frames": 4,             # consecutive loud frames before barge-in while speaking
        "init_coupling": 0.35,         # initial guess of speaker->mic leakage (learned online)
        "speaking_threshold": 0.8,     # stricter VAD probability while she speaks
    },
    "output": {"device": None, "block_ms": 20},
    "lipsync": {"fps": 30, "gain": 1.0},
}


def merge(user: dict | None) -> dict:
    out = copy.deepcopy(DEFAULTS)

    def rec(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                rec(dst[k], v)
            else:
                dst[k] = v

    rec(out, user or {})
    return out
