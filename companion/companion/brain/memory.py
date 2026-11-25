"""Memória local por usuário: janela rolante de conversa + fatos e piadas internas resumidos pelo LLM."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

log = logging.getLogger("companion.memory")

MAX_FACTS = 40
MAX_JOKES = 12
MAX_SUMMARY_CHARS = 700


@dataclass
class Message:
    role: str                 # "user" | "companion"
    text: str
    at: float = 0.0           # epoch
    interrupted: bool = False


@dataclass
class MemoryData:
    facts: list[str] = field(default_factory=list)
    jokes: list[str] = field(default_factory=list)
    summary: str = ""
    recent: list[Message] = field(default_factory=list)
    unsummarized: int = 0     # mensagens ainda não resumidas (as últimas N de `recent`)
    last_seen: float = 0.0


SUMMARIZER_SYSTEM = (
    "Você mantém a memória de longo prazo de uma companheira virtual de desktop. "
    "Recebe a memória atual e as mensagens novas de uma conversa e devolve a memória atualizada. "
    "Responda SOMENTE com um objeto JSON, sem texto em volta, no formato "
    '{"facts": [...], "jokes": [...], "summary": "..."}. '
    "facts: fatos duráveis sobre a pessoa (nome, gostos, rotina, projetos, pessoas importantes, humor atual relevante), "
    "frases curtas, sem duplicar, no máximo 40; descarte o que ficou obsoleto ou foi corrigido. "
    "jokes: piadas internas, apelidos e assuntos recorrentes entre os dois, no máximo 12. "
    "summary: até 600 caracteres sobre o andamento do relacionamento e dos assuntos em aberto. "
    "Não invente nada que não esteja na conversa."
)

Completer = Callable[[str], Awaitable[str]]   # prompt -> texto; o system prompt é SUMMARIZER_SYSTEM


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = " ".join(str(item).split())
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out[-limit:]


def parse_summary_reply(text: str) -> dict | None:
    """Extrai o JSON da resposta do LLM, tolerando cercas de markdown e texto em volta."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class Memory:
    def __init__(self, path: Path, window: int = 30, summarize_after: int = 16, clock: Callable[[], float] = time.time):
        self.path = path
        self.window = window
        self.summarize_after = summarize_after
        self.clock = clock
        self.data = MemoryData()
        self._load()

    # --- persistência -------------------------------------------------
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        d = self.data
        d.facts = [str(x) for x in raw.get("facts", [])]
        d.jokes = [str(x) for x in raw.get("jokes", [])]
        d.summary = str(raw.get("summary", ""))
        d.unsummarized = int(raw.get("unsummarized", 0))
        d.last_seen = float(raw.get("last_seen", 0.0))
        d.recent = [Message(**m) for m in raw.get("recent", []) if isinstance(m, dict) and "role" in m and "text" in m]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.data), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # --- conversa -----------------------------------------------------
    def add(self, role: str, text: str, interrupted: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        d = self.data
        d.recent.append(Message(role, text, self.clock(), interrupted))
        d.recent = d.recent[-self.window:]
        d.unsummarized = min(d.unsummarized + 1, len(d.recent))
        d.last_seen = self.clock()
        self.save()

    def needs_summary(self, min_messages: int | None = None) -> bool:
        return self.data.unsummarized >= (self.summarize_after if min_messages is None else min_messages)

    def seconds_since_seen(self) -> float | None:
        return self.clock() - self.data.last_seen if self.data.last_seen else None

    # --- resumo (fora do caminho quente) ------------------------------
    async def summarize(self, complete: Completer) -> bool:
        """Funde as mensagens novas na memória de longo prazo. Falhar não perde nada: tenta de novo depois."""
        d = self.data
        fresh = d.recent[-d.unsummarized:] if d.unsummarized else []
        if not fresh:
            return False
        count = len(fresh)
        prompt = (
            "MEMÓRIA ATUAL\n" + json.dumps({"facts": d.facts, "jokes": d.jokes, "summary": d.summary}, ensure_ascii=False)
            + "\n\nMENSAGENS NOVAS\n" + render_transcript(fresh, "a pessoa", "a companheira")
        )
        try:
            reply = await complete(prompt)
        except Exception:
            log.exception("resumo da memória falhou")
            return False
        parsed = parse_summary_reply(reply)
        if parsed is None:
            log.warning("resumo da memória veio sem JSON válido")
            return False
        facts = parsed.get("facts", d.facts)
        jokes = parsed.get("jokes", d.jokes)
        d.facts = _dedupe(list(facts) if isinstance(facts, list) else d.facts, MAX_FACTS)
        d.jokes = _dedupe(list(jokes) if isinstance(jokes, list) else d.jokes, MAX_JOKES)
        summary = parsed.get("summary", d.summary)
        d.summary = str(summary)[:MAX_SUMMARY_CHARS]
        d.unsummarized = max(0, d.unsummarized - count)
        self.save()
        return True

    # --- o que entra no prompt ----------------------------------------
    def prompt_block(self, user_label: str, name: str) -> str:
        d = self.data
        parts: list[str] = []
        if d.facts:
            parts.append("O que você sabe dele:\n" + "\n".join(f"- {f}" for f in d.facts))
        if d.jokes:
            parts.append("Piadas internas e assuntos recorrentes:\n" + "\n".join(f"- {j}" for j in d.jokes))
        if d.summary:
            parts.append("Resumo do que já rolou entre vocês: " + d.summary)
        ago = self.seconds_since_seen()
        if ago is not None:
            parts.append("Última conversa: " + humanize_gap(ago) + ".")
        if d.recent:
            parts.append("Últimas mensagens:\n" + render_transcript(d.recent, user_label, name))
        return "\n\n".join(parts)


def render_transcript(messages: list[Message], user_label: str, name: str) -> str:
    lines = []
    for m in messages:
        who = user_label if m.role == "user" else name
        lines.append(f"{who}: {m.text}" + (" [interrompida]" if m.interrupted else ""))
    return "\n".join(lines)


def humanize_gap(seconds: float) -> str:
    if seconds < 90:
        return "agora há pouco"
    if seconds < 3600:
        return f"há {int(seconds // 60)} minutos"
    if seconds < 86400:
        return f"há {int(seconds // 3600)} horas"
    days = int(seconds // 86400)
    return "há 1 dia" if days == 1 else f"há {days} dias"
