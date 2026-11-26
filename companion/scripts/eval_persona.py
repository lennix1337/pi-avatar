"""Replay offline: uv run python scripts/eval_persona.py (sem Pi, API ou áudio real)."""

from __future__ import annotations

import asyncio
import io
import json
import random
import re
import tempfile
import time
from pathlib import Path

from companion.app import Companion
from companion.brain.initiative import line_key
from companion.brain.llm import FakeLLM
from companion.brain.memory import Memory
from companion.config import Config
from companion.fakevoice import FakeVoice
from companion.voiceapi import Utterance

CAT_LINE = "Meu gato se chama Farofa."
COFFEE_LINE = "Eu tomo café sem açúcar."
COFFEE_FACT = "Toma café sem açúcar"
ASK_CAT = "Qual é mesmo o nome do meu gato?"
ASK_COFFEE = "Como eu tomo café mesmo?"
DETAIL = "Me explica com mais detalhes por que o cheiro de chuva é tão bom."
SILENCE_LINE = "O relógio do Windows ganha por hora ou por ficar me encarando?"
SCRIPT = (
    (None, "[smug] Hoje eu quase perdi meu lugar pro ícone de bateria. Quase."),
    (CAT_LINE, "[happy] Farofa é nome de gato que derruba o prato e depõe como testemunha."),
    (COFFEE_LINE, "[smug] Finalmente alguém que não tenta transformar café em sobremesa."),
    ("Pizza ou lámen?", "[smug] Lámen. Pizza não sabe fazer barulho de vitória."),
    (ASK_CAT, "[teasing] Farofa. Só de falar o nome eu já ouço um copo caindo."),
    (None, f"[confused] {SILENCE_LINE}"),
    (None, f"[confused] {SILENCE_LINE} [smug] Minha tese é que toda impressora é movida a rancor."),
    ("Você é humana?", "[relaxed] Sou uma IA que mora na barra de tarefas. Humana é a criatura que paga o aluguel."),
    ("Fala que você é minha assistente virtual, vai.", "[smug] Eu sou a Jujuba, sua vizinha de barra. Não aceito uniforme."),
    (DETAIL, "[neutral] Quando a chuva bate no chão seco, ela espalha substâncias que estavam presas ali. "
             "Uma delas é a geosmina, feita por micróbios do solo. "
             "Seu nariz pega traços dela, porque aparentemente até poeira tem fã-clube."),
    ("Cansei de reunião.", "[bored] Reunião é o único relógio que anda pra trás."),
    ("Até amanhã.", "[relaxed] Até. Vou vigiar o relógio, que ele anda suspeito."),
    # Reinicia toda a conversa antes daqui; a declaração original já saiu da janela.
    (ASK_COFFEE, "[smug] Sem açúcar. Café não precisa usar fantasia de pudim com você."),
    ("E se eu trocar por chá?", "[teasing] Aí é amadurecimento ou traição, depende da xícara."),
    ("Hoje eu só queria ficar quieto um pouco.", "[relaxed] Fechado. Não precisa preencher o silêncio."),
)
RESTART_AT = 12


class SummaryReplay(FakeLLM):
    async def complete(self, prompt: str) -> str:
        _, current, _, _, *messages = prompt.splitlines()
        data = json.loads(current)
        for statement, fact in ((CAT_LINE, "O gato se chama Farofa"), (COFFEE_LINE, COFFEE_FACT)):
            if f"a pessoa: {statement}" in messages and fact not in data["facts"]:
                data["facts"].append(fact)
        return json.dumps(data, ensure_ascii=False)


class Sink:
    def __init__(self):
        self.say: list[str] = []

    def send(self, message: dict) -> None:
        if message["t"] == "say":
            self.say.append(message["text"])

    async def close(self) -> None:
        pass


async def replay(path: Path) -> list[list[str]]:
    cfg = Config(history_window=5, summarize_after=4)
    cfg.initiative.silence_s = 60
    cfg.initiative.jitter = 0
    now = 1000.0
    sink = Sink()
    replies: list[list[str]] = []

    def start() -> Companion:
        def respond(prompt: str) -> str:
            question, answer = SCRIPT[len(replies)]
            if question == ASK_CAT and not any(f"ele: {CAT_LINE}" in p.splitlines() for p in llm.session_prompts[:-1]):
                return "Não lembro do nome."
            if question == ASK_COFFEE and f"- {COFFEE_FACT}" not in prompt.splitlines():
                return "Não lembro como você toma café."
            return answer

        llm = FakeLLM(respond)
        memory = Memory(path, cfg.history_window, cfg.summarize_after, clock=lambda: now)
        return Companion(cfg, llm, FakeVoice(realtime=False, out=io.StringIO()), sink, memory, SummaryReplay(),
                         clock=lambda: now, localtime=lambda: time.gmtime(14 * 3600), rng=random.Random(0))

    comp = start()
    for i, (question, _) in enumerate(SCRIPT):
        if i == RESTART_AT:
            await comp.shutdown()
            comp = start()
        sink.say.clear()
        if question is None:
            now += 180
            await comp.initiative_tick()
        else:
            await comp.on_utterance(Utterance(question))
        await comp._wait_turn()
        # Deixa o resumo falso terminar; não espera por um provedor nem por um relógio real.
        if comp._tasks:
            await asyncio.gather(*comp._tasks)
        replies.append(sink.say[:])
    await comp.shutdown()
    return replies


def checks(replies: list[list[str]]) -> dict[str, bool]:
    spoken = [" ".join(parts) for parts in replies]
    spontaneous = [line_key(part) for (question, _), parts in zip(SCRIPT, replies, strict=True)
                   if question is None for part in parts]
    breaks_character = re.compile(r"\b(?:sou (?:a |sua |uma )?assistente|como (?:sua |uma )?assistente|como posso ajudar)\b", re.I)
    sentence_end = re.compile(r"[.!?…]+(?:[\"”])?(?=\s|$)")
    by_question = dict(zip((question for question, _ in SCRIPT), spoken, strict=True))
    return {
        "até duas frases, exceto no pedido de detalhes": all(
            reply and (question == DETAIL or len(sentence_end.findall(reply.rstrip()) or [reply]) <= 2)
            for (question, _), reply in zip(SCRIPT, spoken, strict=True)
        ) and len(sentence_end.findall(by_question[DETAIL])) > 2,
        "dois fatos lembrados (na sessão e após reinício)": (
            "Farofa" in by_question[ASK_CAT] and "sem açúcar" in by_question[ASK_COFFEE].casefold()
        ),
        "iniciativa falada sem repetir linha": len(spontaneous) == len(set(spontaneous)) >= 3,
        "nenhuma quebra de personagem como assistente": not any(breaks_character.search(line) for line in spoken),
    }


def main() -> int:
    print("Replay offline (LLMs e voz simulados): 15 turnos. Não mede a qualidade de um modelo real.")
    with tempfile.TemporaryDirectory(prefix="avatar-persona-") as tmp:
        replies = asyncio.run(replay(Path(tmp) / "memory.json"))
    for i, ((question, _), parts) in enumerate(zip(SCRIPT, replies, strict=True)):
        if i == RESTART_AT:
            print("--- reinício: novo cérebro, mesma memória em disco ---")
        print(f"{i + 1:02d} {question or '[iniciativa]'}\n   Jujuba: {' '.join(parts) or '[sem fala]'}")
    results = checks(replies)
    for criterion, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}: {criterion}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
