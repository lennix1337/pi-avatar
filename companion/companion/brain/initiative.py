"""Quando ela fala sozinha: silêncio com backoff, madrugada quieta, cutucadas, hora e eventos do harness."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import InitiativeConfig

log = logging.getLogger("companion.initiative")


@dataclass(frozen=True)
class Harness:
    ci_red: bool = False
    merged: bool = False
    workers: int = 0
    waiting_operator: bool = False
    idle_minutes: float = 0.0


def read_harness(path: Path | None) -> Harness | None:
    """Mesmo JSON que o pet lê (ciRed, merged, workers, waitingOperator, idleMinutes). Ausente/ruim → None."""
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return Harness(
            ci_red=bool(raw.get("ciRed", False)),
            merged=bool(raw.get("merged", False)),
            workers=int(raw.get("workers", 0) or 0),
            waiting_operator=bool(raw.get("waitingOperator", False)),
            idle_minutes=float(raw.get("idleMinutes", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Trigger:
    kind: str     # greeting | silence | late | ci_red | merged | waiting | poke
    reason: str   # frase para o contexto do LLM


def time_of_day(hour: int) -> str:
    if hour < 5:
        return "madrugada"
    if hour < 12:
        return "manhã"
    if hour < 18:
        return "tarde"
    return "noite"


_SILENCE_ANGLES = (
    "faça uma observação absurda sobre a vida no desktop",
    "dê uma opinião firme sobre uma bobagem nova",
    "relembre uma deixa leve que ele deu, se houver; senão, faça uma confissão engraçada",
    "invente uma pequena rivalidade sua com um objeto comum",
)


def line_key(text: str) -> str:
    """Compara fala já sem etiquetas, ignorando caixa, espaços e pontuação."""
    return " ".join(re.findall(r"\w+", text.casefold()))


class Initiative:
    def __init__(self, cfg: InitiativeConfig, rng: random.Random | None = None):
        self.cfg = cfg
        self.rng = rng or random.Random()
        self._last_user = float("-inf")       # último sinal dele (fala ou toque)
        self._last_spoke = float("-inf")      # última vez que ela falou (qualquer motivo)
        self._unanswered = 0
        self._next_at: float | None = None
        self._silence_angle = 0
        self._greeted = False
        self._late_day: int | None = None
        self._harness: Harness | None = None
        self._harness_fired: dict[str, float] = {}
        self._last_poke_reply = float("-inf")

    # --- eventos ------------------------------------------------------
    def on_user_activity(self, now: float) -> None:
        self._last_user = now
        self._unanswered = 0
        self._next_at = None

    def on_companion_spoke(self, now: float, initiated: bool) -> None:
        self._last_spoke = now
        if initiated:
            self._unanswered += 1
        self._next_at = None

    @property
    def unanswered(self) -> int:
        return self._unanswered

    def interval(self) -> float:
        base = self.cfg.silence_s * (self.cfg.backoff ** self._unanswered)
        base = min(base, self.cfg.max_interval_s)
        return base * (1 + self.rng.uniform(-self.cfg.jitter, self.cfg.jitter))

    # --- decisões -----------------------------------------------------
    def is_quiet(self, hour: int, now: float) -> bool:
        """Madrugada quieta, a menos que ele esteja conversando."""
        start, end = self.cfg.quiet_start_hour, self.cfg.quiet_end_hour
        night = (start <= hour < end) if start <= end else (hour >= start or hour < end)
        return night and now - self._last_user > self.cfg.active_window_s

    def poke_reply_allowed(self, now: float) -> bool:
        if now - self._last_poke_reply < self.cfg.poke_cooldown_s:
            return False
        self._last_poke_reply = now
        return True

    def check(self, now: float, hour: int, day: int, harness: Harness | None = None) -> Trigger | None:
        """`now` monotônico, `hour`/`day` do relógio local. Chamada a cada segundo mais ou menos."""
        cfg = self.cfg
        if not cfg.enabled:
            return None
        if not self._greeted:
            self._greeted = True
            self._last_spoke = now  # o silêncio conta a partir do cumprimento
            return Trigger("greeting", f"você acabou de acordar no desktop dele; agora é {time_of_day(hour)}")
        quiet = self.is_quiet(hour, now)
        trig = self._harness_trigger(now, harness, quiet)
        if trig:
            return trig
        if quiet:
            return None
        active = now - self._last_user <= cfg.active_window_s
        if active and 1 <= hour < 5 and self._late_day != day:
            self._late_day = day
            return Trigger("late", f"são {hour}h da madrugada e ele ainda está aí conversando com você")
        if self._unanswered >= cfg.max_unanswered:
            return None
        anchor = max(self._last_user, self._last_spoke)
        if self._next_at is None:
            self._next_at = anchor + self.interval()
        if now >= self._next_at:
            # Se a fala falhar ou for descartada, não dispare no próximo tick.
            self._next_at = now + self.interval()
            angle = _SILENCE_ANGLES[self._silence_angle]
            self._silence_angle = (self._silence_angle + 1) % len(_SILENCE_ANGLES)
            return Trigger("silence", f"ninguém falou durante um tempo; {angle}")
        return None

    def _harness_trigger(self, now: float, h: Harness | None, quiet: bool) -> Trigger | None:
        prev, self._harness = self._harness, h
        if h is None or quiet:
            return None
        events: list[tuple[str, str]] = []
        if h.ci_red and not (prev and prev.ci_red):
            events.append(("ci_red", "o CI dele ficou vermelho agora"))
        if h.merged and not (prev and prev.merged):
            events.append(("merged", "um PR dele acabou de ser mesclado"))
        if h.waiting_operator and not (prev and prev.waiting_operator):
            events.append(("waiting", "os agentes dele estão esperando uma resposta dele"))
        for kind, reason in events:
            if now - self._harness_fired.get(kind, float("-inf")) >= self.cfg.harness_cooldown_s:
                self._harness_fired[kind] = now
                return Trigger(kind, reason)
        return None
