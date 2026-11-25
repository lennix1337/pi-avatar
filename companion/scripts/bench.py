"""Mede, com o Pi de verdade, o tempo do fim do texto do usuário até a primeira frase pronta.

uv run python scripts/bench.py --models anthropic/claude-haiku-5-5 anthropic/claude-haiku-4-5 --turns 12
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from companion.brain import persona
from companion.brain.chunker import SentenceChunker
from companion.brain.llm import LLMError, PiRpcLLM
from companion.config import Config

PROMPTS = [
    "oi", "tô com sono hoje", "qual a capital da Austrália?", "você acha que pizza com ketchup é crime?",
    "acabei de terminar um trabalho enorme", "me conta uma coisa aleatória", "kkkk você é muito chata",
    "tava pensando em aprender violão", "o que você faz quando eu fecho o notebook?", "tô meio ansioso com a reunião de amanhã",
    "você prefere café ou chá?", "hoje foi um dia longo", "me dá uma opinião sincera sobre segunda-feira",
]


async def bench(model: str, turns: int, cfg: Config) -> dict:
    llm = PiRpcLLM(cfg.pi_command, model, persona.load_persona(cfg), cfg.thinking, 60.0)
    await llm.start()
    first_token, first_sentence, replies = [], [], []
    try:
        for i in range(turns):
            text = PROMPTS[i % len(PROMPTS)]
            chunker = SentenceChunker()
            t0 = time.perf_counter()
            got_token = got_sentence = None
            out: list[str] = []
            async for delta in llm.stream(f"ele: {text}"):
                if got_token is None:
                    got_token = time.perf_counter() - t0
                out.append(delta)
                if got_sentence is None and chunker.feed(delta):
                    got_sentence = time.perf_counter() - t0
            if got_sentence is None:
                got_sentence = time.perf_counter() - t0
            first_token.append(got_token or 0.0)
            first_sentence.append(got_sentence)
            replies.append((text, "".join(out)))
    finally:
        await llm.close()
    return {"model": model, "tok": first_token, "sent": first_sentence, "replies": replies}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    print("| modelo | turnos | 1o token p50 | 1a frase p50 | 1a frase p90 | max |")
    print("|---|---|---|---|---|---|")
    for model in args.models:
        try:
            r = await bench(model, args.turns, cfg)
        except LLMError as exc:
            print(f"| {model} | falhou: {exc} | | | | |")
            continue
        s = sorted(r["sent"])
        p90 = s[min(len(s) - 1, int(len(s) * 0.9))]
        print(f"| {model} | {len(s)} | {statistics.median(r['tok']):.2f}s | {statistics.median(r['sent']):.2f}s | {p90:.2f}s | {max(s):.2f}s |", flush=True)
        if args.show:
            for q, a in r["replies"]:
                print(f"    ele: {q}\n    ela: {a}")


asyncio.run(main())
