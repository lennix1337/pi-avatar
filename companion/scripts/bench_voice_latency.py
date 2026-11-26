"""Replay the same offline mic → VAD → STT → streamed reply → first PCM on two revisions."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# An editable install can otherwise mask the baseline from a different checkout.
if source := os.environ.get("PI_AVATAR_BENCH_SOURCE"):
    sys.path.insert(0, source)

import companion
from companion.app import Companion
from companion.brain.llm import FakeLLM
from companion.brain.memory import Memory
from companion.config import Config
from companion.server import PetServer
from companion.voice import Voice
from companion.voice.audio import FRAME, MIC_SR
from companion.voice.mic import ArrayMic
from companion.voice.sink import VirtualSink
from companion.voice.tts.fake import FakeTts
from companion.voice.vad import EnergyVad

SENTENCES = ("Tá me ouvindo?", "Olha só que ideia!", "Agora eu entendi.")
MIC_AUDIO = (0.2 * np.sin(2 * np.pi * 240 * np.arange(12 * FRAME) / MIC_SR)).astype(np.float32)


class ReplayLLM(FakeLLM):
    def __init__(self, first: str):
        super().__init__()
        self.first = first

    async def stream(self, prompt):
        await asyncio.sleep(0.070)
        yield self.first
        await asyncio.sleep(0.200)  # punctuation token arrived; next token hasn't yet
        yield " Então vamos continuar."


class FirstPcm(VirtualSink):
    first_at: float | None = None

    def write(self, samples):
        off = super().write(samples)
        if self.first_at is None:
            self.first_at = time.time()
        return off


async def replay(root: Path, sentence: str) -> float:
    sink = FirstPcm(MIC_SR, speed=4)
    mic = ArrayMic(MIC_AUDIO, realtime=True, tail_s=0.7)
    voice = Voice({"stt": {"backend": "fake", "text": "oi"}}, mic=mic, sink=sink,
                  tts=FakeTts(delay=0.035, secs_per_word=0.015), vad=EnergyVad())
    cfg = Config(profile_dir=str(root), filler_after_s=3)
    cfg.initiative.enabled = False
    app = Companion(cfg, ReplayLLM(sentence), voice, PetServer(), Memory(root / "memory.json"))
    try:
        await voice.warmup()
        mic.t0 = time.time()
        utt = await anext(voice.listen())
        await app.on_utterance(utt)
        await asyncio.wait_for(app._wait_turn(), 3)
        if sink.first_at is None:
            raise RuntimeError("replay did not reach the sink")
        return (sink.first_at - utt.ended_at) * 1000
    finally:
        await app.shutdown()


async def measure(rounds: int) -> dict:
    out = {sentence: [] for sentence in SENTENCES}
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(rounds):
            for i, sentence in enumerate(SENTENCES):
                out[sentence].append(await replay(Path(tmp) / str(i), sentence))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3, help="replays per input")
    parser.add_argument("--save", type=Path, help="save raw results for the comparison run")
    parser.add_argument("--before", type=Path, help="raw --save results from the baseline")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    print(f"Companion source: {companion.__file__}")
    data = asyncio.run(measure(args.rounds))
    before = json.loads(args.before.read_text()) if args.before else None
    if before and (before.keys() != data.keys() or any(len(x) != args.rounds for x in before.values())):
        parser.error("--before must use the same fixtures and --rounds")
    print(f"End of mic speech → first PCM queued (ms; median, n={args.rounds} per input)")
    for label in (*SENTENCES, "overall"):
        def median(d):
            return statistics.median(x for xs in d.values() for x in xs) if label == "overall" else statistics.median(d[label])
        prefix = f"before {median(before):.1f} → after " if before else ""
        print(f"{label}: {prefix}{median(data):.1f} ms")
    if args.save:
        args.save.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
