"""`python -m companion`: sobe o servidor do pet, o cérebro e a voz (ou o chat digitado com --text)."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from pathlib import Path

from .app import Companion
from .brain import persona
from .brain.llm import LLM, FakeLLM, PiRpcLLM
from .brain.memory import SUMMARIZER_SYSTEM, Memory
from .config import Config, load_config
from .fakevoice import FakeVoice
from .server import PetServer
from .voiceapi import VoiceIO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="companion", description="Companion do Pi Avatar (cérebro + voz + servidor do pet).")
    p.add_argument("--text", action="store_true", help="conversa digitada no terminal, sem microfone nem áudio")
    p.add_argument("--config", help="arquivo companion.json (padrão: %%APPDATA%%/pi-avatar/companion.json)")
    p.add_argument("--state-file", help="JSON do harness (ciRed, merged, workers, waitingOperator, idleMinutes)")
    p.add_argument("--port", type=int, help="porta TCP do servidor do pet (padrão 47800)")
    p.add_argument("--model", help="modelo do Pi, ex.: anthropic/claude-haiku-5-5")
    p.add_argument("--name", help="nome da companheira")
    p.add_argument("--profile-dir", help="pasta de perfil (config e memória); padrão %%APPDATA%%/pi-avatar")
    p.add_argument("--no-initiative", action="store_true", help="não fala sozinha")
    p.add_argument("--fake-llm", action="store_true", help="respostas de mentira, sem chamar o Pi")
    p.add_argument("--instant", action="store_true", help="a fala digitada não demora (sem tempo de voz simulado)")
    p.add_argument("--doctor", action="store_true", help="confere pi, porta, extra de voz e dispositivos de áudio e sai")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def make_llms(cfg: Config, fake: bool) -> tuple[LLM, LLM | None]:
    if fake:
        return FakeLLM(), FakeLLM()
    chat = PiRpcLLM(cfg.pi_command, cfg.model, persona.load_persona(cfg), cfg.thinking, cfg.llm_timeout_s)
    summarizer = PiRpcLLM(cfg.pi_command, cfg.summarizer_model or cfg.model, SUMMARIZER_SYSTEM, "off", 90.0)
    return chat, summarizer


def create_voice(cfg: Config, text: bool = False, instant: bool = False) -> VoiceIO:
    """A voz real (padrão); o modo digitado (FakeVoice) com --text ou `"text_mode": true` no companion.json."""
    if text or cfg.text_mode:
        return FakeVoice(realtime=not instant)
    try:
        module = importlib.import_module("companion.voice")
    except ImportError as exc:
        sys.exit(f"Voz real indisponível ({exc}). Instale o extra: uv sync --extra voice (ou use --text).")
    return module.create_voice(cfg.voice)


async def _stdin_loop(voice: FakeVoice, name: str) -> None:
    loop = asyncio.get_running_loop()
    print(f"{name} está no ar. Digite e dê Enter; /sair encerra.", flush=True)
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line or line.strip() in ("/sair", "/quit"):
            voice.finish()
            return
        voice.feed(line)


async def amain(args: argparse.Namespace) -> None:
    cfg = load_config(args.config if args.config else (Path(args.profile_dir) / "companion.json" if args.profile_dir else None))
    if args.profile_dir:
        cfg.profile_dir = args.profile_dir
    if args.port:
        cfg.port = args.port
    if args.model:
        cfg.model = args.model
    if args.name:
        cfg.name = args.name
    if args.no_initiative:
        cfg.initiative.enabled = False
    llm, summarizer = make_llms(cfg, args.fake_llm)
    memory = Memory(cfg.memory_dir / "default.json", cfg.history_window, cfg.summarize_after)
    voice = create_voice(cfg, args.text, args.instant)
    server = PetServer(cfg.host, cfg.port)
    harness = Path(args.state_file) if args.state_file else None
    companion = Companion(cfg, llm, voice, server, memory, summarizer, harness)
    reader = asyncio.create_task(_stdin_loop(voice, cfg.name)) if isinstance(voice, FakeVoice) else None
    try:
        await companion.run()
    finally:
        if reader:
            reader.cancel()


def _doctor(args: argparse.Namespace) -> int:
    from .doctor import render, run_checks

    cfg = load_config(args.config if args.config else (Path(args.profile_dir) / "companion.json" if args.profile_dir else None))
    if args.port is not None:
        cfg.port = args.port
    if args.text:
        cfg.text_mode = True
    checks = run_checks(cfg)
    print(render(checks))
    return 0 if all(c.ok for c in checks) else 1


def main() -> None:
    args = build_parser().parse_args()
    if args.doctor:
        sys.exit(_doctor(args))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
