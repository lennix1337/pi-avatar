"""`python -m companion --doctor`: confere o ambiente antes da primeira conversa por voz, sem abrir microfone nem caixa de som."""

from __future__ import annotations

import importlib
import shutil
import socket
from collections.abc import Callable
from dataclasses import dataclass

from .config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _devices() -> list[dict]:
    from .voice.sink import list_devices

    return list_devices()


def run_checks(
    cfg: Config,
    which: Callable[[str], str | None] = shutil.which,
    port_free: Callable[[str, int], bool] = _port_free,
    devices: Callable[[], list[dict]] = _devices,
    importer: Callable[[str], object] = importlib.import_module,
) -> list[Check]:
    checks: list[Check] = []
    pi = which(cfg.pi_command)
    checks.append(Check("pi", bool(pi), pi or f"`{cfg.pi_command}` não está no PATH", "" if pi else "instale o Pi ou ajuste `pi_command` no companion.json"))
    free = port_free(cfg.host, cfg.port)
    checks.append(Check("porta do pet", free, f"{cfg.host}:{cfg.port} " + ("livre" if free else "ocupada"), "" if free else "outra instância já roda? use --port"))
    if cfg.text_mode:
        return checks
    missing = []
    for mod in ("numpy", "sounddevice"):
        try:
            importer(mod)
        except ImportError:
            missing.append(mod)
    checks.append(Check("extra de voz", not missing, "ok" if not missing else "faltando: " + ", ".join(missing), "" if not missing else "uv sync --extra voice (ou use --text)"))
    if missing:
        return checks
    try:
        devs = devices()
    except Exception as exc:  # PortAudio indisponível não pode derrubar o diagnóstico
        checks.append(Check("áudio", False, f"não listou dispositivos: {exc}", "confira o driver de áudio"))
        return checks
    for kind, key, section, field in (
        ("microfone", "max_input_channels", "listen", "input_device"),
        ("caixa de som", "max_output_channels", "output", "device"),
    ):
        chosen = (cfg.voice.get(section) or {}).get(field)
        pool = [d for d in devs if d.get(key, 0) > 0]
        if chosen is not None:
            hit = [d for d in pool if d.get("index") == chosen or d.get("name") == chosen]
            checks.append(Check(kind, bool(hit), hit[0]["name"] if hit else f"dispositivo {chosen!r} não existe", "" if hit else "rode `python -m companion.voice.demo --list-devices` e corrija o índice"))
        else:
            checks.append(Check(kind, bool(pool), f"padrão do sistema ({len(pool)} disponível)" if pool else "nenhum dispositivo", "" if pool else "conecte um headset"))
    return checks


def render(checks: list[Check]) -> str:
    lines = [f"[{'ok' if c.ok else 'FALHOU'}] {c.name}: {c.detail}" + (f"  -> {c.fix}" if c.fix else "") for c in checks]
    lines.append("tudo certo" if all(c.ok for c in checks) else "há itens a corrigir")
    return "\n".join(lines)
