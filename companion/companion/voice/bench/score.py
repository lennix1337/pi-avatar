"""Score a directory of TTS WAVs: round-trip WER through an STT backend and UTMOS.

    python -m companion.voice.bench.score --dir out/kokoro [--stt faster-whisper|canary] [--no-utmos]
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from ..audio import MIC_SR, read_wav_16k
from ..stt import create_stt
from .sentences import SENTENCES


def norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s)


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def utmos(files: list[Path]) -> list[float] | None:
    try:
        try:
            import truststore

            truststore.inject_into_ssl()   # Windows: use the OS certificate store for the hub download
        except ImportError:
            pass
        import torch

        model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    except Exception as e:  # network/hub problems must not hide the WER numbers
        print("UTMOS unavailable:", e)
        return None
    out = []
    for f in files:
        x = torch.from_numpy(read_wav_16k(f)).unsqueeze(0)
        with torch.no_grad():
            out.append(float(model(x, MIC_SR).item()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--stt", default="faster-whisper")
    ap.add_argument("--no-utmos", action="store_true")
    ap.add_argument("--stt-opt", default="{}", help="JSON options for the STT backend")
    a = ap.parse_args()
    files = sorted(a.dir.glob("[0-9][0-9]_*.wav"))
    stt = create_stt({"backend": a.stt, **json.loads(a.stt_opt)})
    stt.load()
    rows = []
    for i, f in enumerate(files):
        hyp = stt.transcribe(read_wav_16k(f))
        rows.append({"file": f.name, "wer": wer(norm(SENTENCES[i]), norm(hyp)), "hyp": hyp})
    mos = None if a.no_utmos else utmos(files)
    for i, r in enumerate(rows):
        r["utmos"] = None if mos is None else mos[i]
    summary = {
        "dir": str(a.dir), "stt": a.stt,
        "wer_mean": round(float(np.mean([r["wer"] for r in rows])), 4),
        "utmos_mean": None if mos is None else round(float(np.mean(mos)), 3),
        "rows": rows,
    }
    (a.dir / "score.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
