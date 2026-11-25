"""Orquestrador: liga voz, cérebro (LLM + memória + iniciativa) e pet, e cuida do ritmo da conversa."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Callable

from . import protocol
from .brain import persona
from .brain.chunker import SentenceChunker
from .brain.initiative import Harness, Initiative, Trigger, read_harness
from .brain.llm import LLM, LLMError
from .brain.memory import Memory
from .brain.tags import Segment, parse_segment
from .config import Config
from .server import PetServer
from .voiceapi import Utterance, VoiceIO

log = logging.getLogger("companion")

FILLERS = ("hmm...", "peraí...", "deixa eu ver...", "então...", "ó...", "hum...")
FAILURES = (
    "[confused] Travei aqui um segundo. Fala de novo?",
    "[embarrassed] Deu um nó na minha cabeça. Repete pra mim?",
    "[bored] Perdi o fio. Culpa da internet. Manda de novo.",
)
POKE_ODDS = {"pat": 0.7, "double": 1.0, "drag": 0.5, "drop": 0.8}
POKE_REACTION = {"pat": ("happy", None), "double": ("surprised", "cheer"), "drag": ("surprised", None),
                 "drop": ("surprised", None)}
INITIATED_KINDS = ("silence", "late")
GAP_FOR_CLOCK_S = 1800.0
USER_TALKING_EXPIRY_S = 15.0


class Companion:
    def __init__(self, cfg: Config, llm: LLM, voice: VoiceIO, server: PetServer, memory: Memory,
                 summarizer: LLM | None = None, harness_file: Path | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 localtime: Callable[[], time.struct_time] = time.localtime,
                 rng: random.Random | None = None):
        self.cfg, self.llm, self.voice, self.server, self.memory = cfg, llm, voice, server, memory
        self.summarizer = summarizer
        self.harness_file = harness_file
        self.clock, self.localtime = clock, localtime
        self.rng = rng or random.Random()
        self.initiative = Initiative(cfg.initiative, self.rng)
        self._turn: asyncio.Task | None = None
        self._said: list[str] = []
        self._partial: str = ""
        self._emotion = "neutral"
        self._state = (False, False, False)
        self._ids = 0
        self._need_seed = True
        self._session_turns = 0
        self._pending_user: str | None = None
        self._interrupt_note: str | None = None
        self._last_user_wall = 0.0
        self._last_band: str | None = None
        self._user_talking_since: float | None = None
        self._last_filler = ""
        self._summarizing: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._harness_mtime = 0.0
        self._harness: Harness | None = None
        self.turn_log: list[dict] = []  # para benchmark e testes

    # --- ciclo de vida ------------------------------------------------
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.server.on_message = self._on_pet_message
        self.server.on_connect = lambda: self._send_state(force=True)
        self.voice.on_speech_start = self._speech_start_threadsafe
        await self.server.start()
        await self.llm.start()
        await self.voice.start()
        self._spawn(self._initiative_loop())
        try:
            async for utt in self.voice.listen():
                await self.on_utterance(utt)
            await self._wait_turn()  # a entrada acabou (stdin fechou): deixa ela terminar a última fala
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        await self._cancel_turn()
        for t in list(self._tasks):
            t.cancel()
        if self.summarizer and self.memory.needs_summary(4):
            try:
                await asyncio.wait_for(self.memory.summarize(self.summarizer.complete), 25)
            except (asyncio.TimeoutError, LLMError):
                log.warning("resumo final não terminou")
        for closer in (self.voice.close, self.llm.close, self.server.close):
            try:
                await closer()
            except Exception:
                log.exception("erro ao fechar")
        if self.summarizer:
            await self.summarizer.close()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # --- estado / envio ----------------------------------------------
    def _send(self, msg: dict) -> None:
        self.server.send(msg)

    def _send_state(self, force: bool = False) -> None:
        if force or self._state:
            self._send(protocol.state(*self._state))

    def _set_state(self, listening: bool | None = None, thinking: bool | None = None, speaking: bool | None = None) -> None:
        cur = self._state
        new = (cur[0] if listening is None else listening, cur[1] if thinking is None else thinking,
               cur[2] if speaking is None else speaking)
        if new != cur:
            self._state = new
            self._send(protocol.state(*new))

    def _on_mouth(self, frame: dict) -> None:
        self._send(protocol.mouth(**{k: frame.get(k, 0.0) for k in protocol.MOUTH_KEYS}))

    def _next_id(self) -> str:
        self._ids += 1
        return f"u{self._ids}"

    @property
    def busy(self) -> bool:
        if self._turn and not self._turn.done():
            return True
        if self._user_talking_since is not None:
            if self.clock() - self._user_talking_since < USER_TALKING_EXPIRY_S:
                return True
            self._user_talking_since = None
        return False

    # --- entrada do usuário ------------------------------------------
    def _speech_start_threadsafe(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.on_user_speech_start)

    def on_user_speech_start(self) -> None:
        """Ele começou a falar: barge-in se ela estiver falando ou pensando."""
        now = self.clock()
        self._user_talking_since = now
        self.initiative.on_user_activity(now)
        if self._turn and not self._turn.done():
            self.voice.cancel()
            self._turn.cancel()
        self._set_state(listening=True, thinking=False, speaking=False)

    async def on_utterance(self, utt: Utterance) -> None:
        text = utt.text.strip()
        if not text:
            return
        await self._join_turn()
        now = self.clock()
        self._user_talking_since = None
        self.initiative.on_user_activity(now)
        context = self._time_context()
        if self._interrupt_note:
            context.append(self._interrupt_note)
            self._interrupt_note = None
        if self._pending_user:
            text = f"{self._pending_user} {text}"
            context.append("ele estava no meio de uma fala, você ainda não tinha respondido, e ele emendou; responda ao conjunto")
            self._pending_user = None
        self._last_user_wall = time.time()
        await self._start_turn("user", persona.user_turn(text, context), user_text=text)

    # --- pet ----------------------------------------------------------
    def _on_pet_message(self, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "hello":
            log.info("pet %s", msg.get("version", "?"))
            self._send_state(force=True)
        elif kind == "poke" and msg.get("kind") in protocol.POKE_KINDS:
            self._spawn(self.on_poke(msg["kind"]))

    async def on_poke(self, kind: str) -> None:
        now = self.clock()
        self.initiative.on_user_activity(now)
        if self._turn and not self._turn.done():
            return
        emotion, gesture = POKE_REACTION[kind]
        self._send(protocol.emotion(emotion, 0.8))
        if gesture:
            self._send(protocol.gesture(gesture))
        if self.rng.random() <= POKE_ODDS[kind] and self.initiative.poke_reply_allowed(now):
            await self._start_turn("poke", persona.poke_turn(kind, self._time_context()))

    # --- iniciativa ---------------------------------------------------
    async def _initiative_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                await self.initiative_tick()
            except Exception:
                log.exception("iniciativa falhou")

    def _read_harness(self) -> Harness | None:
        path = self.harness_file
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        if mtime != self._harness_mtime:
            self._harness_mtime = mtime
            self._harness = read_harness(path)
        return self._harness

    async def initiative_tick(self) -> None:
        if self.busy:
            return
        lt = self.localtime()
        trig = self.initiative.check(self.clock(), lt.tm_hour, lt.tm_yday, self._read_harness())
        if trig:
            await self._start_turn(trig.kind, persona.initiative_turn(trig, self._time_context(force=True)), trigger=trig)

    # --- contexto de tempo -------------------------------------------
    def _time_context(self, force: bool = False) -> list[str]:
        lt = self.localtime()
        band = persona.time_of_day(lt.tm_hour)
        gap = time.time() - self._last_user_wall if self._last_user_wall else None
        out: list[str] = []
        if force or self._last_band != band or gap is None or gap >= GAP_FOR_CLOCK_S:
            line = persona.clock_line(lt.tm_hour, lt.tm_min, lt.tm_wday)
            if gap is not None and gap >= GAP_FOR_CLOCK_S:
                line += f"; ele ficou {int(gap // 60)} minutos sem falar com você"
            out.append(line)
        self._last_band = band
        return out

    # --- turnos -------------------------------------------------------
    async def _join_turn(self) -> None:
        if self._turn and not self._turn.done():
            self.voice.cancel()
            self._turn.cancel()
        await self._wait_turn()

    async def _wait_turn(self) -> None:
        task = self._turn
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _cancel_turn(self) -> None:
        await self._join_turn()

    async def _start_turn(self, kind: str, prompt: str, user_text: str | None = None, trigger: Trigger | None = None) -> None:
        await self._wait_turn()
        self._turn = asyncio.create_task(self._run_turn(kind, prompt, user_text, trigger))

    async def _prepare_session(self, prompt: str) -> str:
        if self._session_turns >= self.cfg.reset_session_after:
            await self.llm.new_session()
            self._need_seed = True
            self._session_turns = 0
        if self._need_seed:
            self._need_seed = False
            lt = self.localtime()
            seed = persona.seed_block(self.memory, self.cfg, persona.clock_line(lt.tm_hour, lt.tm_min, lt.tm_wday))
            prompt = f"{seed}\n{prompt}"
        self._session_turns += 1
        return prompt

    async def _run_turn(self, kind: str, prompt: str, user_text: str | None, trigger: Trigger | None) -> None:
        t0 = self.clock()
        self._said, self._partial = [], ""
        self._set_state(listening=False, thinking=True, speaking=False)
        queue: asyncio.Queue = asyncio.Queue()
        gen: asyncio.Task | None = None
        last_id: str | None = None
        cancelled = False
        failed = False
        try:
            prompt = await self._prepare_session(prompt)
            gen = asyncio.create_task(self._generate(prompt, queue))
            filler_wait = self.cfg.filler_after_s if kind == "user" else None
            first_at: float | None = None
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), filler_wait)
                except asyncio.TimeoutError:
                    filler_wait = None
                    await self._say_filler()
                    continue
                tag, value = item
                if tag == "done":
                    break
                if tag == "error":
                    failed = True
                    log.warning("LLM falhou: %s", value)
                    if not self._said:
                        last_id = await self._say_segment(parse_segment(self.rng.choice(FAILURES)))
                    break
                if first_at is None:
                    first_at = self.clock()
                    self.turn_log.append({"kind": kind, "first_sentence_s": first_at - t0})
                last_id = await self._say_segment(value)
        except asyncio.CancelledError:
            cancelled = True
        finally:
            if gen is not None:
                gen.cancel()
                await asyncio.gather(gen, return_exceptions=True)
            if cancelled:
                self.voice.cancel()
            self._finish_turn(kind, user_text, trigger, last_id, cancelled, failed)

    async def _generate(self, prompt: str, queue: asyncio.Queue) -> None:
        chunker = SentenceChunker()
        try:
            async for delta in self.llm.stream(prompt):
                for raw in chunker.feed(delta):
                    queue.put_nowait(("seg", parse_segment(raw)))
            for raw in chunker.flush():
                queue.put_nowait(("seg", parse_segment(raw)))
            queue.put_nowait(("done", None))
        except LLMError as exc:
            for raw in chunker.flush():
                queue.put_nowait(("seg", parse_segment(raw)))
            queue.put_nowait(("error", str(exc)))

    async def _say_filler(self) -> None:
        choices = [f for f in FILLERS if f != self._last_filler]
        self._last_filler = text = self.rng.choice(choices)
        sid = self._next_id()
        self._send(protocol.say(sid, text, "neutral"))
        self._set_state(thinking=False, speaking=True)
        await self.voice.speak(text, "neutral", self._on_mouth)
        self._set_state(thinking=True, speaking=False)

    async def _say_segment(self, seg: Segment) -> str | None:
        emotion = seg.emotion or self._emotion
        self._emotion = emotion
        for g in seg.gestures:
            self._send(protocol.gesture(g))
        if not seg.text:
            if seg.emotion:
                self._send(protocol.emotion(seg.emotion, 0.8))
            return None
        sid = self._next_id()
        self._send(protocol.say(sid, seg.text, emotion))
        self._set_state(thinking=False, speaking=True)
        self._partial = seg.text
        await self.voice.speak(seg.text, emotion, self._on_mouth)
        self._said.append(seg.text)
        self._partial = ""
        return sid

    def _finish_turn(self, kind: str, user_text: str | None, trigger: Trigger | None, last_id: str | None,
                     cancelled: bool, failed: bool) -> None:
        if last_id:
            self._send(protocol.end(last_id))
        self._set_state(thinking=False, speaking=False)
        spoken = " ".join(self._said)
        if cancelled and self._partial:
            words = self._partial.split()
            spoken = (spoken + " " + " ".join(words[: max(1, len(words) // 2)]) + "—").strip()
        if not spoken and not failed:
            if cancelled and user_text:
                # ele falou por cima antes dela dizer qualquer coisa: a próxima fala se junta a esta
                self._pending_user = user_text
            return
        if user_text:
            self.memory.add("user", user_text)
        if spoken:
            self.memory.add("companion", spoken, interrupted=cancelled)
        if cancelled and spoken:
            self._interrupt_note = f'você foi interrompida; só tinha dito: "{spoken}"'
        self.initiative.on_companion_spoke(self.clock(), initiated=bool(trigger and trigger.kind in INITIATED_KINDS))
        self._maybe_summarize()

    # --- memória ------------------------------------------------------
    def _maybe_summarize(self) -> None:
        if not self.summarizer or not self.memory.needs_summary():
            return
        if self._summarizing and not self._summarizing.done():
            return
        self._summarizing = self._spawn(self.memory.summarize(self.summarizer.complete))
