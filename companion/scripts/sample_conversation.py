"""Conversa de ponta a ponta (orquestrador real + Pi real + voz de mentira) e grava o transcrito em Markdown.

uv run python scripts/sample_conversation.py --model anthropic/claude-haiku-5-5 --out docs/sample-conversation.md
Linhas do roteiro que começam com '>>' interrompem a fala dela (barge-in) em vez de esperar o fim.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from companion.app import Companion
from companion.brain import persona
from companion.brain.llm import PiRpcLLM
from companion.brain.memory import SUMMARIZER_SYSTEM, Memory
from companion.config import Config
from companion.fakevoice import FakeVoice
from companion.voiceapi import Utterance

SCRIPT = [
    "oi, tudo bem?",
    "tô bem, só cansado. passei o dia em reunião",
    "reunião de planejamento, aquelas que podiam ser um e-mail",
    "você acha que um dia as reuniões vão ser substituídas por IA como você?",
    "haha, confiante. mas e se eu te falar que o Alt+Tab me ajuda mais que você?",
    "calma, tô brincando. me ajuda a escolher o jantar: pizza ou lámen?",
    "pizza com abacaxi, fim da discussão",
    "me conta uma curiosidade aleatória",
    ">>espera, deixa eu te contar uma coisa primeiro",
    "é que o meu avô Jorge ficou doente essa semana, e eu tô preocupado",
    "ele tem 84 anos. tá no hospital fazendo uns exames",
    "obrigado. é bom ter com quem falar",
    "bora mudar de assunto. qual seu filme favorito?",
    "você é meio fofa às vezes, sabia?",
    "vou dormir. amanhã eu te conto como foi no hospital",
    "boa noite",
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config().model)
    ap.add_argument("--out", default="docs/sample-conversation.md")
    ap.add_argument("--hour", type=int, default=None, help="força o relógio (hora cheia) para testar horários")
    args = ap.parse_args()
    cfg = Config(model=args.model)
    cfg.initiative.enabled = False
    tmp = Path(tempfile.mkdtemp(prefix="companion-sample-"))
    chat = PiRpcLLM(cfg.pi_command, cfg.model, persona.load_persona(cfg), cfg.thinking, 60.0)
    summarizer = PiRpcLLM(cfg.pi_command, cfg.model, SUMMARIZER_SYSTEM, "off", 90.0)

    class Sink:
        on_message = on_connect = None

        def __init__(self):
            self.say: list[dict] = []

        def send(self, msg):
            if msg["t"] == "say":
                self.say.append(msg)

    sink = Sink()
    voice = FakeVoice(realtime=True, out=open(tmp / "voice.log", "w"))
    memory = Memory(tmp / "memory.json", cfg.history_window, cfg.summarize_after)
    comp = Companion(cfg, chat, voice, sink, memory, summarizer)
    await chat.start()
    comp._loop = asyncio.get_running_loop()
    lines: list[str] = []
    mark = 0

    def flush(note: str = "") -> None:
        nonlocal mark
        spoken = [m["text"] for m in sink.say[mark:]]
        mark = len(sink.say)
        lines.append("**ela:** " + " ".join(spoken) + note + "\n")

    async def finish_turn() -> None:
        while comp._turn and not comp._turn.done():
            await asyncio.sleep(0.02)
        await comp._wait_turn()

    for i, entry in enumerate(SCRIPT):
        interrupt = entry.startswith(">>")
        text = entry.removeprefix(">>")
        if interrupt:
            for _ in range(400):
                if comp._turn and not comp._turn.done() and voice._speaking:
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.5)
            voice.feed(text)  # avisa o início da fala (cancela) e enfileira; consumimos abaixo
            await finish_turn()
            flush(" *[cortada]*")
            voice._queue.get_nowait()
            lines.append("*(ele a interrompe enquanto ela fala)*\n")
        lines.append(f"**ele:** {text}\n")
        await comp.on_utterance(Utterance(text))
        if i + 1 < len(SCRIPT) and SCRIPT[i + 1].startswith(">>"):
            continue
        await finish_turn()
        flush()
    await asyncio.gather(*list(comp._tasks), return_exceptions=True)
    await comp.shutdown()
    firsts = sorted(m["first_sentence_s"] for m in comp.turn_log)
    header = (f"# Conversa de exemplo\n\nModelo: `{args.model}`. Orquestrador real, Pi real, voz simulada; "
              f"{len(firsts)} turnos. Tempo do fim do texto até a primeira frase pronta: p50 {firsts[len(firsts) // 2]:.2f}s, "
              f"máx {firsts[-1]:.2f}s.\n\nMemória extraída ao final (fatos): {memory.data.facts}\n\n---\n\n")
    Path(args.out).write_text(header + "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(header)


asyncio.run(main())
