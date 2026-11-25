import asyncio
import errno
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from companion.config import Config
from companion.doctor import _port_free, render, run_checks


def _mods(missing=()):
    def importer(name):
        if name in missing:
            raise ImportError(name)
        return object()

    return importer


DEVS = [
    {"index": 0, "name": "Mic USB", "max_input_channels": 1, "max_output_channels": 0},
    {"index": 1, "name": "Caixa", "max_input_channels": 0, "max_output_channels": 2},
]


def run(cfg=None, **kw):
    args = dict(which=lambda c: "/bin/pi", port_free=lambda h, p: True, devices=lambda: (DEVS, (0, 1)), importer=_mods())
    args.update(kw)
    return {c.name: c for c in run_checks(cfg or Config(), **args)}


def test_all_green():
    r = run()
    assert all(c.ok for c in r.values()) and render(list(r.values())).endswith("tudo certo")


def test_port_free_checks_if_address_can_be_bound():
    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        assert _port_free(host, port) is False  # Bound, even before listen().
        sock.listen()
        assert _port_free(host, port) is False
    assert _port_free(host, port) is True


def test_port_free_uses_resolved_ipv6_address(monkeypatch):
    from unittest.mock import MagicMock

    address = ("::1", 47800, 0, 0)
    sock = MagicMock()
    sock.__enter__.return_value = sock
    make_socket = MagicMock(return_value=sock)
    monkeypatch.setattr(socket, "socket", make_socket)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address)])
    assert _port_free("::1", 47800) is True
    make_socket.assert_called_once_with(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    sock.setsockopt.assert_any_call(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    sock.bind.assert_called_once_with(address)
    sock.bind.side_effect = OSError(errno.EADDRINUSE, "address in use")
    assert _port_free("::1", 47800) is False


@pytest.mark.parametrize("error", [errno.EADDRNOTAVAIL, 10049], ids=["runtime-errno", "winsock-10049"])
def test_port_free_matches_asyncio_for_unassigned_address(monkeypatch, error):
    host = "127.0.0.1"
    unavailable = "192.0.2.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]

    class ProbeSocket(socket.socket):
        def bind(self, address):
            if address[0] == unavailable:
                raise OSError(error, "address not assigned")
            super().bind(address)

    def resolve(addrs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (addr, port)) for addr in addrs]

    async def asyncio_can_bind(name):
        try:
            server = await asyncio.start_server(lambda r, w: None, name, port, start_serving=False)
        except OSError:
            return False
        server.close()
        await server.wait_closed()
        return True

    monkeypatch.setattr(socket, "socket", ProbeSocket)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: resolve([unavailable, host]))
    assert _port_free("mixed.example", port) == asyncio.run(asyncio_can_bind("mixed.example"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: resolve([unavailable]))
    assert _port_free("unassigned.example", port) is False


@pytest.mark.parametrize("error", [errno.EAFNOSUPPORT, errno.EPERM])
def test_port_free_falls_back_when_ipv6_socket_cannot_be_created(monkeypatch, error):
    info = socket.getaddrinfo("127.0.0.1", 0, type=socket.SOCK_STREAM)
    info.insert(0, (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", 0, 0, 0)))
    create_socket = socket.socket

    def make_socket(family, *args):
        if family == socket.AF_INET6:
            raise OSError(error, "IPv6 unavailable")
        return create_socket(family, *args)

    monkeypatch.setattr(socket, "socket", make_socket)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: info)
    assert _port_free("", 0) is True


def test_port_free_checks_overlapping_addresses_together(monkeypatch):
    from types import SimpleNamespace

    from companion import doctor

    # Exercise the no-reuse configuration with real overlapping binds.
    monkeypatch.setattr(doctor, "os", SimpleNamespace(name="nt"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]

    candidates = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (host, port)) for host in ("0.0.0.0", "127.0.0.1")]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: candidates)
    assert _port_free("overlap.example", port) is False

    # Identical resolver results are not separate binds; the failed probe released its sockets.
    candidates[:] = [candidates[0]] * 2
    assert _port_free("overlap.example", port) is True


def test_port_free_rejects_wildcard_overlap_even_with_reuseaddr(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from companion import doctor

    monkeypatch.setattr(doctor, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(doctor, "sys", SimpleNamespace(platform="linux"))

    def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (addr, port)) for addr in ("127.0.0.1", "0.0.0.0")]

    specific, wildcard = MagicMock(), MagicMock()
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "socket", MagicMock(side_effect=[specific, wildcard, MagicMock(), MagicMock()]))
    assert _port_free("overlap.example", 47800) is False
    specific.bind.assert_called_once_with(("127.0.0.1", 47800))
    wildcard.bind.assert_called_once_with(("0.0.0.0", 47800))
    for sock in (specific, wildcard):
        sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.listen.assert_not_called()
    assert _port_free("overlap.example", 0) is True


def test_port_free_matches_unix_server_restart_after_connection():
    if os.name != "posix" or sys.platform == "cygwin":
        return  # asyncio enables SO_REUSEADDR by default on Unix only.
    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(2)
        client.settimeout(2)
        server.bind((host, 0))
        port = server.getsockname()[1]
        if sys.platform.startswith("linux"):
            # A reusable socket does not prevent a second bind until it listens.
            assert _port_free(host, port) is True
        server.listen()
        assert _port_free(host, port) is False
        client.connect((host, port))
        conn, _ = server.accept()
        with conn:
            conn.settimeout(2)
            conn.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
            client.shutdown(socket.SHUT_WR)
            assert conn.recv(1) == b""
    assert _port_free(host, port) is True


def test_doctor_cli_port_zero_overrides_occupied_configured_port(tmp_path):
    host = "127.0.0.1"
    config = tmp_path / "companion.json"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen()
        config.write_text(
            json.dumps({"host": host, "port": sock.getsockname()[1], "pi_command": sys.executable}),
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "-m", "companion", "--doctor", "--text", "--port", "0", "--config", str(config)],
            capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=10,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[ok] porta do pet: {host}:0 disponível para bind" in result.stdout


def test_doctor_cli_reports_overlong_dns_label(tmp_path):
    host = "a" * 64
    config = tmp_path / "companion.json"
    config.write_text(json.dumps({"host": host, "pi_command": sys.executable}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "companion", "--doctor", "--text", "--config", str(config)],
        capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=10,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"porta do pet: {host}:47800 indisponível" in result.stdout
    assert "Traceback" not in result.stderr


def test_out_of_range_ports_are_rejected():
    assert _port_free("127.0.0.1", -1) is False
    assert _port_free("127.0.0.1", 65536) is False


def test_missing_pi_and_busy_port_have_fixes():
    r = run(which=lambda c: None, port_free=lambda h, p: False)
    assert not r["pi"].ok and "pi_command" in r["pi"].fix
    assert not r["porta do pet"].ok and "--port" in r["porta do pet"].fix


def test_missing_voice_extra_stops_before_devices():
    def boom():
        raise AssertionError("não deve listar dispositivos")

    r = run(importer=_mods({"sounddevice"}), devices=boom)
    assert not r["extra de voz"].ok and "sounddevice" in r["extra de voz"].detail and "áudio" not in r


class PortAudioError(Exception):
    """Stand-in for the exception raised by sounddevice module initialization."""


@pytest.mark.parametrize("error", [OSError("PortAudio library not found"), PortAudioError("Error initializing PortAudio")])
def test_portaudio_import_failure_is_reported(error):
    def importer(name):
        if name == "sounddevice":
            raise error
        return object()

    r = run(importer=importer)
    assert not r["extra de voz"].ok and "PortAudio" in r["extra de voz"].fix
    assert "microfone" not in r and "caixa de som" not in r


def test_listed_devices_do_not_replace_missing_defaults(monkeypatch):
    default = SimpleNamespace(device=(-1, 1))
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(default=default))
    monkeypatch.setitem(sys.modules, "companion.voice.sink", SimpleNamespace(list_devices=lambda: DEVS))

    def check_audio():
        checks = run_checks(Config(), which=lambda _: "/bin/pi", port_free=lambda *_: True, importer=_mods())
        return {c.name: c.ok for c in checks}

    r = check_audio()
    assert not r["microfone"] and r["caixa de som"]
    default.device = (0, -1)
    r = check_audio()
    assert r["microfone"] and not r["caixa de som"]


def test_unknown_configured_device_fails():
    r = run(Config(voice={"listen": {"input_device": 9}, "output": {"device": "Caixa"}}))
    assert not r["microfone"].ok and r["caixa de som"].ok


def test_no_output_device_and_portaudio_error():
    r = run(devices=lambda: (DEVS[:1], (0, -1)))
    assert r["microfone"].ok and not r["caixa de som"].ok

    def err():
        raise OSError("sem PortAudio")

    assert not run(devices=err)["áudio"].ok


def test_text_mode_skips_audio():
    assert set(run(Config(text_mode=True))) == {"pi", "porta do pet"}
