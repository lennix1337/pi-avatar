"""Interface que o orquestrador espera da voz (o pacote companion.voice a implementa) + utterance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Callable

MouthFn = Callable[[dict], None]
FirstAudioFn = Callable[[float], None]  # wall-clock time when the first PCM was queued


@dataclass(frozen=True)
class Utterance:
    text: str
    started_at: float = 0.0
    ended_at: float = 0.0


class VoiceIO(ABC):
    """Ouvir e falar. `on_speech_start` é chamado (de qualquer thread) quando ele começa a falar."""

    on_speech_start: Callable[[], None] | None = None

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    def listen(self) -> AsyncIterator[Utterance]:
        """Falas dele já transcritas, uma por vez."""

    @abstractmethod
    async def speak(self, text: str, emotion: str, on_mouth: MouthFn,
                    on_first_audio: FirstAudioFn | None = None) -> None:
        """Fala `text` e chama `on_mouth` ~30 Hz; `on_first_audio` só é chamado quando há PCM."""

    @abstractmethod
    def cancel(self) -> None:
        """Corta a fala em curso agora."""
