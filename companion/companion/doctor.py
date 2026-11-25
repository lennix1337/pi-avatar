"""`python -m companion --doctor`: confere o ambiente antes da primeira conversa por voz, sem abrir microfone nem caixa de som."""

from __future__ import annotations

import errno
import importlib
import os
import shutil
import socket
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass

from .config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _bind_candidate(stack: ExitStack, family: int, socktype: int, protocol: int, address: tuple) -> bool:
    """Return False for an unusable candidate; propagate other bind failures."""
    try:
        sock = socket.socket(family, socktype, protocol)
    except OSError:
        # asyncio skips candidates it cannot create, even if the OS lists them.
        return False
    stack.enter_context(sock)
    # Match asyncio.start_server's Unix restart and IPv6 defaults.
    if os.name == "posix" and sys.platform != "cygwin":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    try:
        sock.bind(address)
    except OSError as exc:
        # Match asyncio.create_server's errno test on this runtime, not raw Winsock codes.
        if exc.errno == errno.EADDRNOTAVAIL:
            return False
        raise
    return True


def _port_free(host: str, port: int) -> bool:
    """Check whether the server can bind now, without opening a listener."""
    if not 0 <= port <= 65535:
        return False
    try:
        addresses = socket.getaddrinfo(host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except (OSError, UnicodeError):  # Invalid host or failed resolution.
        return False
    wildcard_bound: dict[int, bool] = {}
    try:
        # Keep all unique candidates open until done, so overlapping binds can conflict.
        with ExitStack() as stack:
            for family, kind, proto, _, addr in dict.fromkeys(addresses):
                if not _bind_candidate(stack, family, kind, proto, addr):
                    continue
                wildcard = addr[0] in ("0.0.0.0", "::")
                # SO_REUSEADDR can defer wildcard conflicts until listen on a fixed port.
                if port and family in wildcard_bound and (wildcard or wildcard_bound[family]):
                    return False
                wildcard_bound[family] = wildcard
    except OSError:
        return False
    return bool(wildcard_bound)


def _devices() -> tuple[list[dict], tuple[int, int]]:
    import sounddevice as sd

    from .voice.sink import list_devices

    return list_devices(), (sd.default.device[0], sd.default.device[1])


def run_checks(
    cfg: Config,
    which: Callable[[str], str | None] = shutil.which,
    port_free: Callable[[str, int], bool] = _port_free,
    devices: Callable[[], tuple[list[dict], tuple[int, int]]] = _devices,
    importer: Callable[[str], object] = importlib.import_module,
) -> list[Check]:
    checks: list[Check] = []
    pi = which(cfg.pi_command)
    checks.append(Check("pi", bool(pi), pi or f"`{cfg.pi_command}` não está no PATH", "" if pi else "instale o Pi ou ajuste `pi_command` no companion.json"))
    free = port_free(cfg.host, cfg.port)
    checks.append(Check(
        "porta do pet", free,
        f"{cfg.host}:{cfg.port} " + ("disponível para bind" if free else "indisponível"),
        "" if free else "confira o endereço e a porta (0 a 65535); para trocar a porta, use --port",
    ))
    if cfg.text_mode:
        return checks
    missing = []
    for mod in ("numpy", "sounddevice"):
        try:
            importer(mod)
        except ImportError:
            missing.append(mod)
        except Exception as exc:  # Importing sounddevice also initializes PortAudio.
            fix = "confira a instalação do PortAudio e o driver de áudio" if mod == "sounddevice" else "uv sync --extra voice"
            checks.append(Check("extra de voz", False, f"não carregou {mod}: {exc}", f"{fix} (ou use --text)"))
            return checks
    checks.append(Check("extra de voz", not missing, "ok" if not missing else "faltando: " + ", ".join(missing), "" if not missing else "uv sync --extra voice (ou use --text)"))
    if missing:
        return checks
    try:
        devs, (default_input, default_output) = devices()
    except Exception as exc:  # PortAudio indisponível não pode derrubar o diagnóstico
        checks.append(Check("áudio", False, f"não listou dispositivos: {exc}", "confira o driver de áudio"))
        return checks
    for kind, key, section, field, default_index in (
        ("microfone", "max_input_channels", "listen", "input_device", default_input),
        ("caixa de som", "max_output_channels", "output", "device", default_output),
    ):
        chosen = (cfg.voice.get(section) or {}).get(field)
        pool = [d for d in devs if d.get(key, 0) > 0]
        if chosen is None:
            hit = next((d for d in pool if d.get("index") == default_index), None)
            checks.append(Check(kind, hit is not None, f"padrão: {hit['name']}" if hit else "nenhum dispositivo padrão válido", "" if hit else f"configure o padrão do sistema ou ajuste `{section}.{field}`"))
        else:
            hit = next((d for d in pool if d.get("index") == chosen or d.get("name") == chosen), None)
            checks.append(Check(kind, hit is not None, hit["name"] if hit else f"dispositivo {chosen!r} não existe", "" if hit else "rode `python -m companion.voice.demo --list-devices` e corrija o índice"))
    return checks


def render(checks: list[Check]) -> str:
    lines = [f"[{'ok' if c.ok else 'FALHOU'}] {c.name}: {c.detail}" + (f"  -> {c.fix}" if c.fix else "") for c in checks]
    lines.append("tudo certo" if all(c.ok for c in checks) else "há itens a corrigir")
    return "\n".join(lines)
