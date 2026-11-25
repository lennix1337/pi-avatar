import asyncio
import random

from companion.app import Companion
from companion.brain.llm import FakeLLM, LLMError
from companion.brain.memory import Memory
from companion.config import Config
from companion.fakevoice import FakeVoice
from companion.voiceapi import Utterance


class Sink:
    """Servidor falso: guarda o que iria para o pet."""

    def __init__(self):
        self.sent = []
        self.on_message = None
        self.on_connect = None

    def send(self, msg):
        self.sent.append(msg)

    def types(self):
        return [m["t"] for m in self.sent]

    def of(self, t):
        return [m for m in self.sent if m["t"] == t]


class SlowVoice(FakeVoice):
    """Fala demora `dur` segundos por frase, sem imprimir."""

    def __init__(self, dur=0.3):
        super().__init__(realtime=False)
        self.dur = dur
        self.cancelled = 0

    async def speak(self, text, emotion, on_mouth):
        self.spoken.append((text, emotion))
        self._speaking.add(asyncio.current_task())
        try:
            await asyncio.sleep(self.dur)
        finally:
            self._speaking.discard(asyncio.current_task())

    def cancel(self):
        self.cancelled += 1
        super().cancel()


def build(tmp_path, llm=None, voice=None, **cfg_kw):
    cfg = Config(profile_dir=str(tmp_path), **cfg_kw)
    cfg.initiative.enabled = False
    llm = llm or FakeLLM(["[smug] Primeira frase aqui. [gesture:tilt] [teasing] Segunda frase também."])
    voice = voice or SlowVoice(0.01)
    sink = Sink()
    mem = Memory(tmp_path / "m.json", cfg.history_window, cfg.summarize_after)
    comp = Companion(cfg, llm, voice, sink, mem, summarizer=llm, rng=random.Random(3))
    comp._loop = asyncio.get_running_loop()
    return comp, llm, voice, sink, mem


async def settle(comp):
    while comp._turn and not comp._turn.done():
        await asyncio.sleep(0.01)
    await comp._wait_turn()


async def say(comp, text):
    await comp.on_utterance(Utterance(text))
    await settle(comp)


async def test_reply_is_split_into_say_emotion_gesture_end_and_tags_are_never_spoken(tmp_path):
    comp, llm, voice, sink, mem = build(tmp_path)
    await say(comp, "oi")
    says = sink.of("say")
    assert [s["text"] for s in says] == ["Primeira frase aqui.", "Segunda frase também."]
    assert [s["emotion"] for s in says] == ["smug", "teasing"]
    assert sink.of("gesture") == [{"t": "gesture", "name": "tilt"}]
    assert sink.of("end") == [{"t": "end", "id": says[-1]["id"]}]
    assert all("[" not in t for t, _ in voice.spoken)
    states = [(m["thinking"], m["speaking"]) for m in sink.of("state")]
    assert states[0] == (True, False) and (False, True) in states and states[-1] == (False, False)
    assert [m.role for m in mem.data.recent] == ["user", "companion"]


async def test_first_prompt_carries_memory_seed_and_clock_next_ones_do_not(tmp_path):
    comp, llm, *_ = build(tmp_path)
    await say(comp, "oi")
    await say(comp, "e aí")
    assert "[contexto: sessão nova" in llm.prompts[0] and "ele: oi" in llm.prompts[0]
    assert "sessão nova" not in llm.prompts[1] and llm.prompts[1].endswith("ele: e aí")


async def test_barge_in_cancels_speech_and_generation_and_keeps_interrupted_text(tmp_path):
    llm = FakeLLM(["Primeira frase bem comprida aqui. Segunda frase que nunca sai. Terceira."], chunk_delay=0.01)
    voice = SlowVoice(0.3)
    comp, llm, voice, sink, mem = build(tmp_path, llm=llm, voice=voice)
    await comp.on_utterance(Utterance("conta uma história"))
    for _ in range(100):
        if voice.spoken:
            break
        await asyncio.sleep(0.01)
    assert voice.spoken
    comp.on_user_speech_start()                       # ele começa a falar por cima
    await settle(comp)
    assert voice.cancelled >= 1
    assert llm.aborted == 1
    assert [t for t, _ in voice.spoken] == ["Primeira frase bem comprida aqui."]
    assert sink.of("state")[-1]["listening"] is True and sink.of("state")[-1]["speaking"] is False
    saved = [m for m in mem.data.recent if m.role == "companion"]
    assert saved and saved[0].interrupted and "Primeira frase" in saved[0].text
    await say(comp, "foi mal, continua")
    assert "você foi interrompida" in llm.prompts[-1] and "Primeira" in llm.prompts[-1]


async def test_barge_in_before_any_speech_merges_both_user_messages(tmp_path):
    llm = FakeLLM(["Resposta qualquer pra isso aqui."], first_delay=0.3)
    comp, llm, voice, sink, mem = build(tmp_path, llm=llm)
    await comp.on_utterance(Utterance("estava pensando em"))
    await asyncio.sleep(0.05)
    comp.on_user_speech_start()
    await settle(comp)
    assert voice.spoken == [] and mem.data.recent == []
    llm.first_delay = 0
    await say(comp, "comprar um violão")
    assert "estava pensando em comprar um violão" in llm.prompts[-1]


async def test_filler_only_when_first_token_is_late(tmp_path):
    comp, llm, voice, sink, mem = build(tmp_path, llm=FakeLLM(first_delay=0.0), filler_after_s=0.1)
    await say(comp, "rápido")
    assert len(voice.spoken) == 2                      # sem filler
    (tmp_path / "b").mkdir()
    comp, llm, voice, sink, mem = build(tmp_path / "b", llm=FakeLLM(first_delay=0.35), filler_after_s=0.1)
    await say(comp, "devagar")
    assert len(voice.spoken) == 3
    assert voice.spoken[0][0].endswith("...") and voice.spoken[0][1] == "neutral"
    assert [m.role for m in mem.data.recent] == ["user", "companion"]
    assert not mem.data.recent[1].text.startswith(voice.spoken[0][0])    # o filler não entra na memória


async def test_llm_failure_gets_a_spoken_apology_and_does_not_poison_history(tmp_path):
    class Broken(FakeLLM):
        async def stream(self, prompt):
            raise LLMError("sem rede")
            yield ""

    comp, llm, voice, sink, mem = build(tmp_path, llm=Broken())
    await say(comp, "oi")
    assert len(voice.spoken) == 1 and sink.of("end")


async def test_pet_pokes_trigger_emotion_gesture_and_a_reply(tmp_path):
    comp, llm, voice, sink, mem = build(tmp_path, llm=FakeLLM(["[embarrassed] Ei! Avisa antes, tá?"]))
    comp.rng = random.Random(0)
    comp._on_pet_message({"t": "poke", "kind": "double"})
    await asyncio.sleep(0.05)
    await settle(comp)
    assert sink.of("gesture")[0]["name"] == "cheer"
    assert "[toque: double]" in llm.prompts[0]
    assert sink.of("say")
    comp._on_pet_message({"t": "poke", "kind": "pat"})   # dentro do cooldown: gesto sim, fala não
    await asyncio.sleep(0.05)
    await settle(comp)
    assert len(llm.prompts) == 1 and sink.of("emotion")


async def test_hello_from_pet_gets_current_state(tmp_path):
    comp, *_, sink, _m = build(tmp_path)
    comp._on_pet_message({"t": "hello", "version": "1.0"})
    assert sink.types() == ["state"]


async def test_initiative_tick_speaks_on_its_own_and_harness_events_reach_the_prompt(tmp_path):
    comp, llm, voice, sink, mem = build(tmp_path, llm=FakeLLM(["[happy] Oi, sumido!"]))
    comp.cfg.initiative.enabled = True
    comp.initiative.cfg.enabled = True
    comp.localtime = lambda: __import__("time").struct_time((2025, 5, 5, 14, 0, 0, 0, 125, 0))
    await comp.initiative_tick()
    await settle(comp)
    assert "[iniciativa:" in llm.prompts[0] and sink.of("say")
    assert not comp.initiative.unanswered


async def test_summary_runs_off_the_hot_path_after_enough_turns(tmp_path):
    comp, llm, voice, sink, mem = build(tmp_path, summarize_after=4)
    llm.completion_reply = '{"facts": ["gosta de violão"], "jokes": [], "summary": "papo de música"}'
    for text in ("oi", "tudo certo?"):
        await say(comp, text)
    await asyncio.gather(*list(comp._tasks), return_exceptions=True)
    assert mem.data.facts == ["gosta de violão"] and len(llm.completions) == 1


async def test_session_resets_after_n_turns_and_reseeds_memory(tmp_path):
    comp, llm, *_ = build(tmp_path, reset_session_after=2)
    for t in ("a", "b", "c"):
        await say(comp, t)
    assert llm.sessions == 2
    assert "sessão nova" in llm.prompts[2]


async def test_run_lets_the_last_reply_finish_when_input_ends(tmp_path):
    comp, llm, voice, sink, mem = build(tmp_path)
    comp.server = _FakeServer(sink)
    voice.feed("oi")
    voice.finish()
    await asyncio.wait_for(comp.run(), 5)
    assert len(voice.spoken) == 2 and sink.of("end")


class _FakeServer:
    def __init__(self, sink):
        self.sink = sink
        self.on_message = self.on_connect = None

    async def start(self):
        return None

    async def close(self):
        return None

    def send(self, msg):
        self.sink.send(msg)


class PipeVoice(FakeVoice):
    """Voz que toca uma frase por vez (como a real) e mede quantas estão na fila ao mesmo tempo."""

    def __init__(self, dur=0.15):
        super().__init__(realtime=False)
        self.dur, self.inflight, self.peak, self.events = dur, 0, 0, []
        self.gate = asyncio.Lock()

    async def speak(self, text, emotion, on_mouth):
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self.events.append(("submit", text))
        try:
            async with self.gate:
                self.events.append(("start", text))
                await asyncio.sleep(self.dur)
                self.events.append(("end", text))
        finally:
            self.inflight -= 1


async def test_next_sentence_is_queued_while_the_previous_plays_but_shown_when_it_starts(tmp_path):
    llm = FakeLLM(["Primeira frase. Segunda frase. Terceira frase. Quarta frase."])
    comp, llm, voice, sink, mem = build(tmp_path, llm=llm, voice=PipeVoice(0.15))
    seen = {}
    orig_end = voice.speak

    async def spy(text, emotion, on_mouth):
        await orig_end(text, emotion, on_mouth)
        seen[text] = [m["text"] for m in sink.of("say")]    # legendas já enviadas quando esta frase acabou

    voice.speak = spy
    await say(comp, "conta")
    kinds = [(k, t) for k, t in voice.events]
    assert kinds.index(("submit", "Segunda frase.")) < kinds.index(("end", "Primeira frase."))   # síntese adiantada, sem buraco entre frases
    assert voice.peak == 2                                                  # mas só uma frase de folga
    assert seen["Primeira frase."] == ["Primeira frase."]                                          # legenda da segunda só quando ela começa a soar
    assert [m["text"] for m in sink.of("say")] == ["Primeira frase.", "Segunda frase.", "Terceira frase.", "Quarta frase."]
    assert sink.of("end") == [{"t": "end", "id": sink.of("say")[-1]["id"]}]
    assert mem.data.recent[-1].text == "Primeira frase. Segunda frase. Terceira frase. Quarta frase." and not mem.data.recent[-1].interrupted


async def test_barge_in_with_a_sentence_queued_keeps_only_what_was_audible(tmp_path):
    llm = FakeLLM(["Primeira frase longa. Segunda frase. Terceira frase."])
    comp, llm, voice, sink, mem = build(tmp_path, llm=llm, voice=PipeVoice(0.3))
    await comp.on_utterance(Utterance("conta"))
    for _ in range(100):
        if ("start", "Primeira frase longa.") in voice.events:
            break
        await asyncio.sleep(0.01)
    comp.on_user_speech_start()
    await settle(comp)
    assert [m["text"] for m in sink.of("say")] == ["Primeira frase longa."]
    saved = [m for m in mem.data.recent if m.role == "companion"]
    assert saved and saved[0].interrupted and "Segunda" not in saved[0].text
    assert voice.inflight == 0


async def test_a_failing_voice_does_not_kill_the_turn(tmp_path):
    class Broken(PipeVoice):
        async def speak(self, text, emotion, on_mouth):
            raise RuntimeError("tts caiu")

    comp, llm, voice, sink, mem = build(tmp_path, voice=Broken())
    await say(comp, "oi")
    assert sink.of("state")[-1]["speaking"] is False
