"""Speech-to-text backends. `transcribe(audio16k) -> str` is blocking: callers run it in a thread."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


class SttBackend:
    name = "base"

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeStt(SttBackend):
    """Returns canned text, for tests."""

    name = "fake"

    def __init__(self, text: str | list[str] = "olá", **_):
        self.texts = [text] if isinstance(text, str) else list(text)
        self.calls: list[int] = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(audio.size)
        return self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]


def add_cuda_dll_dirs() -> None:
    """Windows: ctranslate2 finds cuBLAS/cuDNN from the pip `nvidia-*` wheels only if their bin dirs are registered."""
    if os.name != "nt":
        return
    import site
    import sys

    for root in {*site.getsitepackages(), *(p for p in sys.path if p.endswith("site-packages"))}:
        for d in Path(root, "nvidia").glob("*/bin"):
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


class FasterWhisperStt(SttBackend):
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", compute_type: str = "float16",
                 language: str = "pt", beam_size: int = 1, **_):
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self.language, self.beam_size = language, beam_size
        self.model = None

    def load(self) -> None:
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(
            np.asarray(audio, np.float32), language=self.language, beam_size=self.beam_size,
            vad_filter=False, condition_on_previous_text=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segs).strip()


class CanaryStt(SttBackend):
    """Pi Voice's STT: Canary 1B v2 (Q8_0 GGUF) on transcribe.cpp, driven through a Node sidecar."""

    name = "canary"

    def __init__(self, model_path: str = "", node: str = "node", node_modules: str = "",
                 language: str = "pt", backend: str = "auto", **_):
        self.model_path = model_path or os.environ.get("COMPANION_CANARY_MODEL", "")
        self.node = node
        self.node_modules = node_modules or os.environ.get("COMPANION_TRANSCRIBE_MODULES", "")
        self.language = language
        self.backend = backend
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.last_ms: int | None = None   # engine time of the last request

    def load(self) -> None:
        if not (self.model_path and self.node_modules):
            raise RuntimeError("canary needs stt.canary.model_path and stt.canary.node_modules")
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [self.node, str(HERE / "canary_worker.mjs"), self.node_modules, self.model_path, self.language,
             self.backend],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err)
        ready = json.loads(self.proc.stdout.readline() or b"{}")
        if not ready.get("ready"):
            self._err.seek(0)
            raise RuntimeError(f"canary sidecar failed to start: {ready} {self._err.read().decode(errors='replace')[-800:]}")

    def transcribe(self, audio: np.ndarray) -> str:
        pcm = np.ascontiguousarray(audio, dtype="<f4")
        with self._lock:
            self.proc.stdin.write(struct.pack("<I", pcm.size) + pcm.tobytes())
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("canary sidecar closed")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(r["error"])
        self.last_ms = r.get("ms")
        return r["text"]

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None


def create_stt(cfg: dict) -> SttBackend:
    name = cfg.get("backend", "faster-whisper")
    common = {k: cfg[k] for k in ("language",) if k in cfg}
    if name == "faster-whisper":
        return FasterWhisperStt(**{**common, **{k: cfg[k] for k in ("model", "device", "compute_type", "beam_size") if k in cfg}})
    if name == "canary":
        return CanaryStt(**{**common, **cfg.get("canary", {})})
    if name == "fake":
        return FakeStt(**{k: v for k, v in cfg.items() if k == "text"})
    raise ValueError(f"unknown STT backend {name!r}")
