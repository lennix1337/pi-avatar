"""Carrega a persona (embutida ou de um arquivo do usuário) e monta os prompts de cada turno."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from ..config import Config
from .initiative import Trigger, time_of_day
from .memory import Memory

_WEEKDAYS = ("segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo")


def load_persona(cfg: Config) -> str:
    if cfg.persona_path:
        text = Path(cfg.persona_path).read_text(encoding="utf-8")
    else:
        text = resources.files("companion.brain").joinpath("persona.md").read_text(encoding="utf-8")
    user = cfg.user_name or "quem usa este computador"
    return text.replace("{name}", cfg.name).replace("{user}", user)


def user_label(cfg: Config) -> str:
    return "ele"


def clock_line(hour: int, minute: int, weekday: int) -> str:
    return f"{hour:02d}:{minute:02d}, {time_of_day(hour)} de {_WEEKDAYS[weekday]}"


def seed_block(memory: Memory, cfg: Config, now_line: str) -> str:
    """Primeiro prompt de cada sessão do Pi: memória e contexto, para ela 'lembrar'."""
    block = memory.prompt_block(user_label(cfg), "você")
    head = f"[contexto: sessão nova. Agora: {now_line}.]"
    if block:
        return f"{head}\n[memória, só para você; não recite]\n{block}\n[fim da memória]"
    return f"{head}\n[memória vazia: vocês ainda não se conhecem direito]"


def user_turn(text: str, context: list[str]) -> str:
    lines = [f"[contexto: {c}]" for c in context]
    lines.append(f"ele: {text}")
    return "\n".join(lines)


def initiative_turn(trigger: Trigger, context: list[str]) -> str:
    lines = [f"[contexto: {c}]" for c in context]
    lines.append(f"[iniciativa: {trigger.reason}]")
    return "\n".join(lines)


def poke_turn(kind: str, context: list[str]) -> str:
    lines = [f"[contexto: {c}]" for c in context]
    lines.append(f"[toque: {kind}]")
    return "\n".join(lines)
