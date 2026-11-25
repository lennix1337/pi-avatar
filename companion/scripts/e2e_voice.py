"""Conversa de ponta a ponta sem microfone: voz real (Canary + Supertonic), cérebro real (Pi) e o pet por TCP.

    uv run --extra voice python scripts/e2e_voice.py --out docs/e2e [--model anthropic/claude-sonnet-5-5]

As falas dele são sintetizadas (voz M1 do Supertonic), entram no `listen()` por um microfone injetado em tempo real
(VAD Silero + STT Canary de verdade) e a resposta dela sai por uma placa de som virtual com relógio real: nada toca
nos alto-falantes. Um gravador conecta no servidor do pet (127.0.0.1:47800) como o pet faria; o pet de verdade pode
estar conectado ao mesmo tempo. Grava transcrito, latências por turno, WAVs das respostas e os quadros enviados ao pet.
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
from companion.voice import Voice
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
MOUTH_KEYS = ("aa", "ih", "ou", "ee", "oh")


class InjectMic(Mic):
    """Microfone de teste: silêncio em tempo real; o áudio empurrado por `push()` sai quadro a quadro, no relógio."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self._pending: deque[tuple[deque, dict]] = deque()
        self._closed = False

    def push(self, audio: np.ndarray) -> dict:
        x = np.concatenate([audio.astype(np.float32), np.zeros((-audio.size) % FRAME, np.float32)])
        blocks = x.reshape(-1, FRAME)
        level = np.sqrt(np.mean(np.square(blocks), axis=1))
        voiced = np.flatnonzero(level > 0.1 * level.max())   # o TTS deixa silêncio antes e depois da voz
        mark: dict = {"start": None, "end": None, "first": int(voiced[0]), "last": int(voiced[-1])}
        self._pending.append((deque(blocks), mark))
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
                cur = self._pending.popleft()
                cur[1]["i"] = 0
            if cur is None:
                yield silence
                continue
            frame, mark = cur[0].popleft(), cur[1]
            if mark["i"] == mark["first"]:
                mark["start"] = time.perf_counter() - FRAME_S    # primeiro quadro com voz
            if mark["i"] == mark["last"]:
                mark["end"] = time.perf_counter()                 # último quadro com voz
            mark["i"] += 1
            if not cur[0]:
                cur = None
            yield frame

    async def close(self) -> None:
        self._closed = True


class RecordingSink(VirtualSink):
    """Placa de som virtual que guarda o que "tocou", por turno."""

    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self.turn = 0
        self.chunks: dict[int, list[np.ndarray]] = {}

    def write(self, samples: np.ndarray) -> float:
        self.chunks.setdefault(self.turn, []).append(np.array(samples, np.float32))
        return super().write(samples)


class Probe(Voice):
    """A voz real com carimbos de tempo nas bordas que o orquestrador usa."""

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
            if call["t_mouth"] is None and max((w.get(k, 0.0) for k in MOUTH_KEYS), default=0.0) > 0.05:
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
    for _ in range(100):
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


def first_audio_of(voice: Probe, call: dict) -> float | None:
    """Instante em que o primeiro áudio da frase entrou na saída (histórico do Speaker, casado por texto e ordem)."""
    for item in voice._speaker.history:
        if item.text == call["text"] and abs(item.t_submit - call["t_submit"]) < 0.05 and item.first_audio_s is not None:
            return item.t_submit + item.first_audio_s
    return None


def f(v: float | None, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


async def amain(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(model=args.model, port=args.port, profile_dir=tempfile.mkdtemp(prefix="companion-e2e-"))
    cfg.initiative.enabled = False

    t = time.perf_counter()
    user_tts = create_tts("supertonic", voice="M1")
    user_tts.load()
    said = [*LINES, STORY, INTERRUPT]
    user_audio = {s: synth(user_tts, s) for s in said}
    for i, s in enumerate(said, 1):
        write_wav(out / f"user-{i}.wav", user_audio[s], MIC_SR)
    print(f"falas dele sintetizadas em {time.perf_counter() - t:.1f}s", flush=True)

    tts = create_tts("supertonic")
    mic = InjectMic()
    # a voz carrega o TTS dentro do start(); a placa virtual precisa da taxa de amostragem, que só existe depois do load()
    tts.load()
    sink = RecordingSink(tts.sample_rate)
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
    await wait_for(lambda: bool(voice._tasks) or run.done(), 180, "listen() não começou")
    if run.done():
        run.result()
    boot_s = time.perf_counter() - t_boot
    print(f"no ar em {boot_s:.1f}s (STT + Pi)", flush=True)
    peak_clients = 0
    turns: list[dict] = []

    async def say(text: str, kind: str, until_done: bool = True) -> dict:
        nonlocal peak_clients
        n = len(voice.utts) + 1
        prev = comp._turn
        mark = mic.push(user_audio[text])
        turn = {"turn": n, "kind": kind, "said": text, "mark": mark}
        turns.append(turn)
        await wait_for(lambda: len(voice.utts) >= n, 90, f"fala {n} não transcrita")
        if until_done:
            await asyncio.sleep(0.3)
            await wait_for(lambda: comp._turn is not prev and comp._turn is not None and comp._turn.done()
                           and not voice.speaking, 120, f"turno {n} não terminou")
            await asyncio.sleep(1.0)
        peak_clients = max(peak_clients, len(server._clients))
        return turn

    for line in LINES:
        await say(line, "normal")

    # barge-in: ele interrompe quando ela já está contando a história
    story = await say(STORY, "story-interrupted", until_done=False)
    await wait_for(lambda: any(c["turn"] == story["turn"] and not c["filler"] and c["t_mouth"] for c in voice.calls), 90,
                   "ela não começou a história")
    await asyncio.sleep(2.5)
    interrupt_turn_before = len(voice.cancels)
    await say(INTERRUPT, "interruption")
    story["interrupted"] = len(voice.cancels) > interrupt_turn_before

    stop.set()
    await mic.close()
    await asyncio.wait_for(run, 120)
    await pet
    report(out, args, turns, voice, sink, pet_frames, boot_s, memory, peak_clients)


def report(out: Path, args, turns, voice: Probe, sink, pet_frames, boot_s, memory, peak_clients) -> None:
    rows = []
    for t in turns:
        n, mark = t["turn"], t["mark"]
        utt = voice.utts[n - 1]
        calls = [c for c in voice.calls if c["turn"] == n]
        real = [c for c in calls if not c["filler"]]
        first = real[0] if real else None
        fa_first = first_audio_of(voice, first) if first else None
        fa_any = min((x for x in (first_audio_of(voice, c) for c in calls) if x), default=None)
        done = max((c["t_done"] for c in calls if c["t_done"]), default=None)
        rows.append({
            "turn": n, "kind": t["kind"], "said": t["said"], "heard": utt["text"],
            "speech_end_to_text_s": utt["t_text"] - mark["end"],
            "stt_s": utt["stt_s"],
            "text_to_first_sentence_s": first["t_submit"] - utt["t_text"] if first else None,
            "first_sentence_to_first_audio_s": fa_first - first["t_submit"] if fa_first else None,
            "speech_end_to_first_audio_s": fa_any - mark["end"] if fa_any else None,
            "first_audio_to_mouth_s": first["t_mouth"] - fa_first if first and first["t_mouth"] and fa_first else None,
            "total_s": done - mark["end"] if done else None,
            "reply_audio_s": sum(c.size for c in sink.chunks.get(n, [])) / sink.sample_rate,
            "filler": [c["text"] for c in calls if c["filler"]],
            "sentences": [c["text"] for c in real],
        })
    bi = None
    last = turns[-1]["mark"]
    starts = [s for s in voice.speech_starts if s >= last["start"] - 0.05]
    cancels = [c for c in voice.cancels if c >= last["start"] - 0.05]
    if starts and cancels:
        bi = {"user_voice_to_speech_confirmed_s": round(starts[0] - last["start"], 3),
              "user_voice_to_audio_cut_s": round(cancels[0] - last["start"], 3)}
    (out / "latency.json").write_text(json.dumps({"boot_s": boot_s, "model": args.model, "turns": rows, "barge_in": bi},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# Conversa de ponta a ponta (voz real, Pi real, pet por TCP)", "",
             f"Modelo: `{args.model}`. Entrada: Supertonic M1 → microfone injetado em tempo real → VAD Silero → Canary. "
             "Saída: Supertonic F1 → placa virtual (relógio real, sem alto-falante). PC com RTX 4070 SUPER.", ""]
    for r in rows:
        lines += [f"## Turno {r['turn']} ({r['kind']})", f"- Ele disse: {r['said']}", f"- Ouvido (STT): {r['heard']}"]
        if r["filler"]:
            lines.append(f"- Enrolação: {', '.join(r['filler'])}")
        lines += ["- Ela: " + " ".join(r["sentences"]), ""]
    if bi:
        lines += ["## Interrupção (barge-in)",
                  f"- Ele começou a falar → fala confirmada pelo VAD: {bi['user_voice_to_speech_confirmed_s']} s",
                  f"- Ele começou a falar → áudio dela cortado: {bi['user_voice_to_audio_cut_s']} s", ""]
    (out / "transcript.md").write_text("\n".join(lines), encoding="utf-8")

    tab = ["| Turno | Fim da fala → texto | (só o STT) | Texto → 1ª frase do LLM | 1ª frase → 1º áudio | Fim da fala → 1º áudio | Total (fim da fala → fim do áudio) | Áudio da resposta |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tab.append(f"| {r['turn']} ({r['kind']}) | {f(r['speech_end_to_text_s'])} s | {f(r['stt_s'], 3)} s | {f(r['text_to_first_sentence_s'])} s | "
                   f"{f(r['first_sentence_to_first_audio_s'])} s | {f(r['speech_end_to_first_audio_s'])} s | {f(r['total_s'])} s | {f(r['reply_audio_s'], 1)} s |")
    normal = [r for r in rows if r["kind"] == "normal"]
    med = {k: statistics.median(v) for k in ("speech_end_to_text_s", "text_to_first_sentence_s", "first_sentence_to_first_audio_s",
                                             "speech_end_to_first_audio_s", "total_s")
           if (v := [r[k] for r in normal if r[k] is not None])}
    tab.append(f"| **mediana (turnos normais)** | {f(med.get('speech_end_to_text_s'))} s | | {f(med.get('text_to_first_sentence_s'))} s | "
               f"{f(med.get('first_sentence_to_first_audio_s'))} s | {f(med.get('speech_end_to_first_audio_s'))} s | {f(med.get('total_s'))} s | |")
    (out / "latency.md").write_text("\n".join(tab) + "\n", encoding="utf-8")

    for turn, chunks in sink.chunks.items():
        write_wav(out / f"reply-turn-{turn}.wav", np.concatenate(chunks), sink.sample_rate)
    kinds: dict[str, int] = {}
    for m in pet_frames:
        kinds[m["t"]] = kinds.get(m["t"], 0) + 1
    t0 = pet_frames[0]["at"] if pet_frames else 0.0
    (out / "pet-frames.jsonl").write_text("\n".join(json.dumps({**m, "at": round(m["at"] - t0, 3)}, ensure_ascii=False)
                                                   for m in pet_frames), encoding="utf-8")
    summary = {"pet_messages": kinds, "mouth_frames_open": sum(1 for m in pet_frames if m["t"] == "mouth" and max(
        (m.get(k, 0) for k in MOUTH_KEYS), default=0) > 0.05), "peak_pet_clients": peak_clients,
        "barge_in": bi, "boot_s": round(boot_s, 1), "reply_wavs": sorted(f"reply-turn-{t}.wav" for t in sink.chunks)}
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
