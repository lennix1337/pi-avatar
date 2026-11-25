"""Mensagens JSON-por-linha entre o companion (servidor) e o pet. Contrato em companion/PROTOCOL.md."""

from __future__ import annotations

import json
from typing import Any

EMOTIONS = (
    "neutral", "happy", "angry", "sad", "relaxed", "surprised",
    "smug", "teasing", "embarrassed", "bored", "excited", "confused",
)
GESTURES = (
    "wave", "cheer", "nod", "shake", "tilt", "laugh", "shrug", "lean_in", "look_away",
)
POKE_KINDS = ("pat", "double", "drag", "drop")
MOUTH_KEYS = ("aa", "ih", "ou", "ee", "oh")


def encode(msg: dict[str, Any]) -> bytes:
    return (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict[str, Any] | None:
    """Uma linha do pet; devolve None para o que for inválido (descartado sem derrubar a conexão)."""
    try:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        msg = json.loads(text)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("t"), str):
        return None
    return msg


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 3)


def state(listening: bool, thinking: bool, speaking: bool) -> dict[str, Any]:
    return {"t": "state", "listening": listening, "thinking": thinking, "speaking": speaking}


def say(id: str, text: str, emotion: str = "neutral") -> dict[str, Any]:
    return {"t": "say", "id": id, "text": text, "emotion": emotion if emotion in EMOTIONS else "neutral"}


def mouth(aa: float = 0.0, ih: float = 0.0, ou: float = 0.0, ee: float = 0.0, oh: float = 0.0) -> dict[str, Any]:
    return {"t": "mouth", "aa": _clamp(aa), "ih": _clamp(ih), "ou": _clamp(ou), "ee": _clamp(ee), "oh": _clamp(oh)}


def emotion(name: str, intensity: float = 0.8) -> dict[str, Any]:
    return {"t": "emotion", "name": name if name in EMOTIONS else "neutral", "intensity": round(_clamp(intensity), 2)}


def gesture(name: str) -> dict[str, Any]:
    return {"t": "gesture", "name": name}


def end(id: str) -> dict[str, Any]:
    return {"t": "end", "id": id}
