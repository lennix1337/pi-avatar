"""Configuração do companion: arquivo JSON no perfil do usuário, com padrões neutros."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_profile_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "pi-avatar"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "pi-avatar"


@dataclass
class InitiativeConfig:
    enabled: bool = True
    silence_s: float = 300.0       # silêncio até a primeira fala espontânea
    backoff: float = 2.0           # cada fala sem resposta multiplica o intervalo
    max_interval_s: float = 3600.0
    max_unanswered: int = 3        # depois disso, só volta quando ele falar
    jitter: float = 0.3            # ±30% no intervalo
    quiet_start_hour: int = 0      # madrugada quieta, salvo se ele estiver conversando
    quiet_end_hour: int = 7
    active_window_s: float = 600.0 # "conversando" = falou nos últimos 10 min
    harness_cooldown_s: float = 600.0
    poke_cooldown_s: float = 20.0


@dataclass
class Config:
    name: str = "Jujuba"
    user_name: str = ""
    host: str = "127.0.0.1"
    port: int = 47800
    pi_command: str = "pi"
    model: str = "anthropic/claude-sonnet-5-5"
    summarizer_model: str = ""      # vazio = o mesmo modelo da conversa
    thinking: str = "off"
    persona_path: str = ""          # vazio = persona embutida
    profile_dir: str = ""           # vazio = %APPDATA%/pi-avatar
    history_window: int = 30        # mensagens na janela rolante
    summarize_after: int = 16       # mensagens novas que disparam o resumo
    reset_session_after: int = 60   # turnos até recomeçar a sessão do Pi com memória fresca
    filler_after_s: float = 1.2     # LLM mais lento que isso → "hmm", "peraí"
    llm_timeout_s: float = 40.0
    text_mode: bool = False         # true = conversa digitada (como --text), sem microfone nem áudio
    voice: dict = field(default_factory=dict)   # sobrepõe os padrões de companion/voice/config.py (tts, stt, listen, output...)
    initiative: InitiativeConfig = field(default_factory=InitiativeConfig)

    @property
    def profile(self) -> Path:
        return Path(self.profile_dir) if self.profile_dir else default_profile_dir()

    @property
    def memory_dir(self) -> Path:
        return self.profile / "memory"


def _merge(obj: Any, data: dict[str, Any]) -> None:
    for f in dataclasses.fields(obj):
        if f.name not in data:
            continue
        cur, new = getattr(obj, f.name), data[f.name]
        if dataclasses.is_dataclass(cur) and isinstance(new, dict):
            _merge(cur, new)
        elif isinstance(new, type(cur)) or (isinstance(cur, float) and isinstance(new, int) and not isinstance(new, bool)):
            setattr(obj, f.name, new)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else cfg.profile / "companion.json"
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _merge(cfg, data)
    return cfg
