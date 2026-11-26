"""Corta o texto que chega em pedaços do LLM em frases prontas para falar, o mais cedo possível."""

from __future__ import annotations

import re

# !/? já fecham uma frase no fim de um token; ponto espera contexto (abreviações, decimais).
_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s|\[)|[!?]+[\"')\]]*$|\n+")
_TAG_ONLY = re.compile(r"\[[^\[\]]*\]")
MIN_CHARS = 10   # frase mais curta que isso (sem etiquetas) espera a próxima, p.ex. "Hm." + "Então..."
MAX_CHARS = 160  # sem pontuação por tanto tempo: corta numa vírgula


def _visible(s: str) -> str:
    return _TAG_ONLY.sub("", s).strip()


class SentenceChunker:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            sentence = self._take()
            if sentence is None:
                break
            out.append(sentence)
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if _visible(rest) else []

    def _take(self) -> str | None:
        buf = self._buf
        for m in _END.finditer(buf):
            end = m.end()
            # uma etiqueta aberta ("[smug") no meio da frase não pode ser cortada
            candidate = buf[:end]
            if candidate.count("[") != candidate.count("]"):
                continue
            if len(_visible(candidate)) < MIN_CHARS and not self._exclaims(candidate):
                continue
            self._buf = buf[end:].lstrip(" \t")
            return candidate.strip()
        if len(buf) > MAX_CHARS:
            cut = max(buf.rfind(", ", 0, MAX_CHARS), buf.rfind("; ", 0, MAX_CHARS), buf.rfind(" — ", 0, MAX_CHARS))
            if cut > 40:
                self._buf = buf[cut + 1:].lstrip()
                return buf[: cut + 1].strip()
        return None

    @staticmethod
    def _exclaims(candidate: str) -> bool:
        # "Ué?" e "Não!" são curtas mas completas; só ponto simples ("Hm.") espera
        return bool(re.search(r"[!?]", _visible(candidate)))
