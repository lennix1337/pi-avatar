from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .lipsync import resample

MIC_SR = 16000          # everything on the listening side runs at 16 kHz mono
FRAME = 512             # 32 ms at 16 kHz: Silero's window


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def read_wav_16k(path: str | Path) -> np.ndarray:
    x, sr = read_wav(path)
    return resample(x, sr, MIC_SR)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-14)) if len(x) else 0.0


def dbfs(level: float) -> float:
    return 20.0 * float(np.log10(level + 1e-9))
