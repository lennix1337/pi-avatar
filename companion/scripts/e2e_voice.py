"""Conversa de ponta a ponta sem microfone: voz real (Supertonic + Canary), cérebro real (Pi) e um "pet" gravador.

    uv run --extra voice python scripts/e2e_voice.py --out docs/e2e [--model anthropic/claude-sonnet-5-5]

As falas dele são sintetizadas (voz M1 do Supertonic), entram no `listen()` por um microfone injetado em tempo real
e a resposta dela sai por uma placa de som virtual (relógio real, nada toca nos alto-falantes). O gravador conecta no
servidor do pet (127.0.0.1:47800) como o pet faria. Grava transcrito, latências por turno, os WAVs e os quadros do pet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

from companion.__main__ import make_llms
from companion.app import FILLERS, Companion
from companion.brain.memory import Memory
from companion.config import Config
from companion.server import PetServer
from companion.voice import Utterance, Voice
from companion.voice.audio import FRAME, MIC_SR, write_wav
from companion.voice.lipsync import resample
from companion.voice.mic import FRAME_S, Mic
from companion.voice.sink import VirtualSink
from companion.voice.tts import create as create_tts

LINES = [
    "Oi, tudo bem? Acordei agora e já quero conversar com você.",
    "Hoje eu passei o dia inteiro em reunião, tô exausto.",
    "Você acha que eu devo jantar pizza ou lámen?",
    "Me conta uma curiosidade aleatória sobre o espaço.",
    "Valeu, você é bem divertida. Agora preciso dormir, boa noite!",
]
STORY = "Me conta uma história bem comprida sobre um dragão que tinha medo de fogo."
INTERRUPT = "Espera, para um segundo, deixa eu te falar uma coisa."


class InjectMic(Mic):
    """Microfone de teste: silêncio em tempo real, e o áudio empurrado por `push()` quando houver."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self._pending: deque[tuple[np.ndarray, dict]] = deque()
        self._closed = False

    def push(self, audio: np.ndarray) -> dict:
        pad = (-audio.size) % FRAME
        x = np.concatenate([audio.astype(np.float32), np.zeros(pad, np.float32)])
        mark = {"start": None, "end": None}
        self._pending.append((x.reshape(-1, FRAME), mark))
        return mark

    async def frames(self):
        self.t0 = time.time()
        start, k = time.monotonic(), 0
        cur: tuple[deque, dict] | None = None
        silence = np.zeros(FRAME, np.float32)
        while not self._closed:
            k += 1
            await asyncio.sleep(max(0.0, start + k * FRAME_S - time.monotonic()))
            if cur is None and self._pending:
                blocks, mark = self._pending.popleft()
                cur = (deque(blocks), mark)
                mark["start"] = time.perf_counter() - FRAME_S
            if cur is None:
                yield silence
                continue
            frame = cur[0].popleft()
            if not cur[0]:
                cur[1]["end"] = time.perf_counter()
                cur = None
            yield frame

    async def close(self) -> None:
        self._closed = True


class RecordingSink(VirtualSink):
    """Placa de som virtual que guarda tudo o que "tocou", por turno."""

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self.turn = 0
        self.chunks: dict[int, list[np.ndarray]] = {}

    def write(self, samples: np.ndarray) -> float:
        self.chunks.setdefault(self.turn, []).append(np.array(samples, np.float32))
        return super().write(samples)


class Probe(Voice):
    """Voice real com carimbos de tempo nas bordas que o orquestrador usa."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.utts: list[dict] = []
        self.calls: list[dict] = []
        self.cancels: list[float] = []
        self.speech_starts: list[float] = []

    async def listen(self):
        async for utt in super().listen():
            self.utts.append({"text": utt.text, "t_text": time.perf_counter(), "stt_s": self.stats["stt_s"][-1]})
            self._sink.turn = len(self.utts)
            yield utt

    async def speak(self, text, emotion, on_mouth):
        call = {"turn": len(self.utts), "text": text, "emotion": emotion, "t_submit": time.perf_counter(),
                "t_mouth": None, "t_done": None, "filler": text in FILLERS}
        self.calls.append(call)

        def mouth(w: dict) -> None:
            if call["t_mouth"] is None and max(w.values(), default=0.0) > 0.05:
                call["t_mouth"] = time.perf_counter()
            on_mouth(w)

        try:
            await super().speak(text, emotion, mouth)
        finally:
            call["t_done"] = time.perf_counter()

    def cancel(self) -> None:
        self.cancels.append(time.perf_counter())
        super().cancel()

    def _fire_start(self) -> None:
        self.speech_starts.append(time.perf_counter())
        super()._fire_start()


async def pet_recorder(port: int, frames: list[dict], stop: asyncio.Event) -> None:
    for _ in range(50):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            break
        except OSError:
            await asyncio.sleep(0.1)
    else:
        raise RuntimeError("servidor do pet não subiu")
    while not stop.is_set():
        try:
            line = await asyncio.wait_for(reader.readline(), 0.5)
        except asyncio.TimeoutError:
            continue
        if not line:
            break
        frames.append({"at": time.perf_counter(), **json.loads(line)})
    writer.close()


def synth(tts, text: str) -> np.ndarray:
    x = np.concatenate([c for c in tts.stream(text, "neutral") if c.size])
    return resample(x, tts.sample_rate, MIC_SR)


async def wait_for(pred, timeout: float, what: str) -> None:
    t = time.perf_counter()
    while not pred():
        if time.perf_counter() - t > timeout:
            raise TimeoutError(what)
        await asyncio.sleep(0.02)


async def amain(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(model=args.model, port=args.port, profile_dir=tempfile.mkdtemp(prefix="companion-e2e-"))
    cfg.initiative.enabled = False

    t = time.perf_counter()
    user_tts = create_tts("supertonic", voice="M1")
    user_tts.load()
    user_audio = {s: synth(user_tts, s) for s in [*LINES, STORY, INTERRUPT]}
    user_tts.close()
    for i, s in enumerate([*LINES, STORY, INTERRUPT], 1):
        write_wav(out / f"user-{i}.wav", user_audio[s], MIC_SR)
    print(f"falas dele sintetizadas em {time.perf_counter() - t:.1f}s", flush=True)

    tts = create_tts("supertonic")
    tts.load()
    sink = RecordingSink(tts.sample_rate)
    mic = InjectMic()
    voice = Probe({}, mic=mic, tts=tts, sink=sink)
    llm, summarizer = make_llms(cfg, False)
    memory = Memory(cfg.memory_dir / "default.json", cfg.history_window, cfg.summarize_after)
    server = PetServer(cfg.host, cfg.port)
    comp = Companion(cfg, llm, voice, server, memory, summarizer)

    pet_frames: list[dict] = []
    stop = asyncio.Event()
    t_boot = time.perf_counter()
    run = asyncio.create_task(comp.run())
    pet = asyncio.create_task(pet_recorder(cfg.port, pet_frames, stop))
    await wait_for(lambda: voice._speaker is not None or voice._stt is not None, 120, "modelos")
    await wait_for(lambda: bool(voice._tasks), 120, "listen() não começou")
    boot_s = time.perf_counter() - t_boot
    print(f"no ar em {boot_s:.1f}s (modelos + Pi)", flush=True)

    turns: list[dict] = []

    async def settle(n_utts: int, timeout: float = 120.0) -> None:
        await wait_for(lambda: len(voice.utts) >= n_utts, timeout, f"fala {n_utts} não transcrita")
        await asyncio.sleep(0.2)
        await wait_for(lambda: comp._turn is not None and comp._turn.done() and not voice.speaking, timeout,
                       f"turno {n_utts} não terminou")
        await asyncio.sleep(1.0)

    for line in LINES:
        n = len(voice.utts) + 1
        mark = mic.push(user_audio[line])
        await settle(n)
        turns.append({"turn": n, "kind": "normal", "said": line, "mark": mark})

    # barge-in: ele interrompe no meio da história
    n = len(voice.utts) + 1
    mark = mic.push(user_audio[STORY])
    await settle(n)  # a história inteira; depois o turno de interrupção abaixo cai num turno novo
    turns.append({"turn": n, "kind": "normal", "said": STORY, "mark": mark})
    n = len(voice.utts) + 1
    mark = mic.push(user_audio[STORY])
    await wait_for(lambda: any(c["turn"] == n and not c["filler"] and c["t_done"] is None for c in voice.calls), 90,
                   "ela não começou a história")
    await asyncio.sleep(2.0)                      # deixa ela falar um pouco
    mark2 = mic.push(user_audio[INTERRUPT])
    turns.append({"turn": n, "kind": "story-interrupted", "said": STORY, "mark": mark})
    await wait_for(lambda: len(voice.utts) >= n + 1, 60, "interrupção não transcrita")
    await settle(n + 1)
    turns.append({"turn": n + 1, "kind": "interruption", "said": INTERRUPT, "mark": mark2})

    stop.set()
    await mic.close()
    await asyncio.wait_for(run, 120)
    await pet

    report(out, args, turns, voice, sink, comp, pet_frames, boot_s, memory)


def f(v: float | None, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def report(out, args, turns, voice, sink, comp, pet_frames, boot_s, memory) -> None:
    hist = {(i.text): i for i in voice._speaker.history}
    rows, md = [], []
    for t in turns:
        n, mark = t["turn"], t["mark"]
        utt = voice.utts[n - 1]
        calls = [c for c in voice.calls if c["turn"] == n]
        real = [c for c in calls if not c["filler"]]
        first = real[0] if real else None
        item = hist.get(first["text"]) if first else None
        first_audio = first["t_submit"] + item.first_audio_s if item and item.first_audio_s is not None else None
        first_any = min((c["t_submit"] + hist[c["text"]].first_audio_s for c in calls
                         if c["text"] in hist and hist[c["text"]].first_audio_s is not None), default=None)
        done = max((c["t_done"] for c in calls if c["t_done"]), default=None)
        tl = next((x for x in comp.turn_log if False), None)  # placeholder: turn_log é por ordem, abaixo
        row = {
            "turn": n, "kind": t["kind"], "said": t["said"], "heard": utt["text"],
            "speech_end_to_text_s": utt["t_text"] - mark["end"], "stt_s": utt["stt_s"],
            "text_to_first_sentence_s": first["t_submit"] - utt["t_text"] if first else None,
            "first_sentence_to_first_audio_s": item.first_audio_s if item else None,
            "text_to_first_audio_s": first_audio - utt["t_text"] if first_audio else None,
            "speech_end_to_first_audio_s": (first_any - mark["end"]) if first_any else None,
            "first_audio_to_mouth_s": (first["t_mouth"] - first_audio) if first and first["t_mouth"] and first_audio else None,
            "total_s": (done - mark["end"]) if done else None,
            "filler": [c["text"] for c in calls if c["filler"]],
            "sentences": [c["text"] for c in real],
        }
        rows.append(row)
    for r in rows:
        md.append(r)
    (out / "latency.json").write_text(json.dumps({"boot_s": boot_s, "model": args.model, "turns": rows,
                                                   "starved_s": sink.starved_s}, ensure_ascii=False, indent=1), encoding="utf-8")

    # barge-in
    bi = None
    if voice.cancels and voice.speech_starts:
        mark = next(t["mark"] for t in turns if t["kind"] == "story-interrupted")
        # o 2º push (a interrupção) é o último "speech start" antes do último cancel
        interrupt_mark = turns[-1]["mark"]
        start_after = [s for s in voice.speech_starts if s >= interrupt_mark["start"]]
        cancel_after = [c for c in voice.cancels if c >= interrupt_mark["start"]]
        if start_after and cancel_after:
            bi = {"user_start_to_confirmed_s": start_after[0] - interrupt_mark["start"],
                  "user_start_to_cancel_s": cancel_after[0] - interrupt_mark["start"]}

    # transcrito
    lines = ["# Conversa de ponta a ponta (voz real, Pi real, pet gravador)", "",
             f"Modelo: `{args.model}`. Entrada: Supertonic M1 → microfone injetado em tempo real → VAD Silero → Canary. "
             "Saída: Supertonic F1 → placa virtual. Latências medidas no PC (RTX 4070 SUPER).", ""]
    for r in rows:
        lines += [f"## Turno {r['turn']} ({r['kind']})", f"- Ele disse: {r['said']}", f"- Ouvido (STT): {r['heard']}"]
        if r["filler"]:
            lines.append(f"- Enrolação: {', '.join(r['filler'])}")
        lines.append("- Ela: " + " ".join(r["sentences"]))
        lines.append("")
    if bi:
        lines += ["## Interrupção (barge-in)", f"- Ele começou a falar → fala confirmada: {f(bi['user_start_to_confirmed_s'])} s",
                  f"- Ele começou a falar → áudio dela cortado: {f(bi['user_start_to_cancel_s'])} s", ""]
    (out / "transcript.md").write_text("\n".join(lines), encoding="utf-8")

    tab = ["| Turno | Fim da fala → texto (s) | (só STT) | Texto → 1ª frase do LLM (s) | 1ª frase → 1º áudio (s) | Fim da fala → 1º áudio (s) | Total (s) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        tab.append(f"| {r['turn']} {r['kind']} | {f(r['speech_end_to_text_s'])} | {f(r['stt_s'], 3)} | {f(r['text_to_first_sentence_s'])} | "
                   f"{f(r['first_sentence_to_first_audio_s'])} | {f(r['speech_end_to_first_audio_s'])} | {f(r['total_s'])} |")
    normal = [r for r in rows if r["kind"] == "normal"]
    for key, label in (("speech_end_to_text_s", "fim da fala → texto"), ("text_to_first_sentence_s", "texto → 1ª frase"),
                       ("first_sentence_to_first_audio_s", "1ª frase → 1º áudio"), ("speech_end_to_first_audio_s", "fim da fala → 1º áudio")):
        vals = [r[key] for r in normal if r[key] is not None]
        if vals:
            tab.append(f"| **mediana** {label} | {statistics.median(vals):.2f} | | | | | |")
    (out / "latency.md").write_text("\n".join(tab) + "\n", encoding="utf-8")

    # áudio das respostas e quadros do pet
    for turn, chunks in sink.chunks.items():
        write_wav(out / f"reply-turn-{turn}.wav", np.concatenate(chunks), sink.sample_rate)
    types: dict[str, int] = {}
    for m in pet_frames:
        types[m["t"]] = types.get(m["t"], 0) + 1
    t0 = pet_frames[0]["at"] if pet_frames else 0.0
    (out / "pet-frames.jsonl").write_text(
        "\n".join(json.dumps({**m, "at": round(m["at"] - t0, 3)}, ensure_ascii=False) for m in pet_frames), encoding="utf-8")
    summary = {"pet_messages": types, "mouth_nonzero": sum(1 for m in pet_frames if m["t"] == "mouth" and max(
        (m.get(k, 0) for k in ("aa", "ih", "ou", "ee", "oh")), default=0) > 0.05), "starved_s": round(sink.starved_s, 3),
        "barge_in": bi, "boot_s": round(boot_s, 1), "memory_lines": len(memory.data.recent)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(tab))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/e2e")
    ap.add_argument("--model", default=Config().model)
    ap.add_argument("--port", type=int, default=47800)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
