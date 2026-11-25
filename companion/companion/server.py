"""Servidor TCP (127.0.0.1:47800) do companion: o pet conecta e troca JSON por linha."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from . import protocol

log = logging.getLogger("companion.server")
PetHandler = Callable[[dict[str, Any]], None]
MAX_BUFFERED = 64 * 1024  # acima disso, quadros de boca atrasados são descartados


class PetServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 47800, on_message: PetHandler | None = None,
                 on_connect: Callable[[], None] | None = None):
        self.host, self.port = host, port
        self.on_message = on_message
        self.on_connect = on_connect
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._tasks: set[asyncio.Task] = set()

    @property
    def connected(self) -> bool:
        return bool(self._clients)

    @property
    def bound_port(self) -> int:
        assert self._server is not None and self._server.sockets
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._client, self.host, self.port)
        log.info("servidor do pet em %s:%s", self.host, self.bound_port)

    async def close(self) -> None:
        if self._server:
            self._server.close()
        for w in list(self._clients):
            w.close()
        for t in list(self._tasks):
            t.cancel()
        if self._server:
            await self._server.wait_closed()

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task:
            self._tasks.add(task)
        self._clients.add(writer)
        log.info("pet conectado")
        if self.on_connect:
            self.on_connect()
        try:
            while True:
                try:
                    line = await reader.readline()
                except ValueError:
                    continue  # linha maior que o limite: descartada
                if not line:
                    break
                msg = protocol.decode(line)
                if msg is None:
                    log.debug("mensagem inválida do pet descartada")
                    continue
                if self.on_message:
                    try:
                        self.on_message(msg)
                    except Exception:
                        log.exception("erro tratando %s", msg.get("t"))
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self._clients.discard(writer)
            if task:
                self._tasks.discard(task)
            writer.close()
            log.info("pet desconectou")

    def send(self, msg: dict[str, Any]) -> None:
        """Envia a todos os pets conectados. Sem pet, é um no-op."""
        if not self._clients:
            return
        data = protocol.encode(msg)
        for w in list(self._clients):
            if w.is_closing():
                self._clients.discard(w)
                continue
            if msg.get("t") == "mouth" and w.transport.get_write_buffer_size() > MAX_BUFFERED:
                continue
            try:
                w.write(data)
            except (ConnectionError, RuntimeError):
                self._clients.discard(w)
