"""Etiquetas inline da persona: `[smug]` (emoção) e `[gesture:tilt]` (gesto). Saem do texto falado."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..protocol import EMOTIONS, GESTURES

_TAG = re.compile(r"\[\s*([A-Za-z_]+)\s*(?::\s*([A-Za-z_]+)\s*)?\]")
# qualquer colchete curto que sobrar (ex.: "[risos]") também não deve ser lido em voz alta
_STRAY = re.compile(r"\[[^\[\]\n]{1,30}\]")
_ALIASES = {"laughs": "laugh", "giggle": "laugh", "smirk": "smug", "tease": "teasing",
            "blush": "embarrassed", "shy": "embarrassed", "sleepy": "bored", "thinking": "confused",
            "sigh": "bored", "lean": "lean_in", "lookaway": "look_away"}


@dataclass
class Segment:
    """Uma frase falada, já sem etiquetas, com a emoção e os gestos que ela pediu."""
    text: str
    emotion: str | None = None
    gestures: list[str] = field(default_factory=list)


def parse_segment(raw: str) -> Segment:
    emotion: str | None = None
    gestures: list[str] = []

    def take(match: re.Match[str]) -> str:
        nonlocal emotion
        head, tail = match.group(1).lower(), (match.group(2) or "").lower()
        head = _ALIASES.get(head, head)
        tail = _ALIASES.get(tail, tail)
        if head in ("gesture", "gesto") and tail in GESTURES:
            gestures.append(tail)
        elif head in ("emotion", "emocao", "emoção") and tail in EMOTIONS:
            emotion = emotion or tail
        elif not tail and head in EMOTIONS:
            emotion = emotion or head
        elif not tail and head in GESTURES:
            gestures.append(head)
        return " "

    text = _TAG.sub(take, raw)
    text = _STRAY.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return Segment(text=text, emotion=emotion, gestures=gestures)
