"""LLM só pelo Pi: um processo `pi --mode rpc` persistente, sem ferramentas, com a persona como system prompt."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Callable

log = logging.getLogger("companion.llm")


class LLMError(RuntimeError):
    pass


class LLM(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Pedaços de texto da resposta. Fechar o gerador (ou cancelar a task) aborta a geração."""

    @abstractmethod
    async def new_session(self) -> None:
        """Esquece o contexto do Pi; o chamador reenvia a memória no primeiro prompt."""

    async def complete(self, prompt: str) -> str:
        await self.new_session()
        return "".join([chunk async for chunk in self.stream(prompt)])


class PiRpcLLM(LLM):
    def __init__(self, command: str | list[str], model: str, system_prompt: str, thinking: str = "off",
                 timeout: float = 40.0, workdir: Path | None = None):
        self.command = command
        self.model = model
        self.system_prompt = system_prompt
        self.thinking = thinking
        self.timeout = timeout
        self._tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="pi-companion-"))
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._sink: asyncio.Queue | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()
        self._seq = 0
        self.first_token_s: float | None = None  # medida do último prompt, para o benchmark

    # --- processo -----------------------------------------------------
    def argv(self) -> list[str]:
        self._tmp.mkdir(parents=True, exist_ok=True)
        prompt_file = self._tmp / "persona.md"
        prompt_file.write_text(self.system_prompt, encoding="utf-8")
        base = list(self.command) if isinstance(self.command, list) else [shutil.which(self.command) or self.command]
        args = [*base, "--mode", "rpc", "--no-session", "--no-tools", "--no-extensions", "--no-skills",
                "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-mcp", "--offline",
                "--model", self.model, "--thinking", self.thinking,
                "--system-prompt", str(prompt_file)]
        return args

    async def start(self) -> None:
        if self._proc and self._proc.returncode is None:
            return
        self._tmp.mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            *self.argv(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, cwd=self._tmp, limit=1 << 24,
        )
        self._reader = asyncio.create_task(self._read_loop(self._proc))
        self._idle.set()

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if self._reader:
            self._reader.cancel()
        if proc and proc.returncode is None:
            try:
                assert proc.stdin is not None
                proc.stdin.close()
                await asyncio.wait_for(proc.wait(), 3)
            except (asyncio.TimeoutError, OSError):
                proc.kill()
                await proc.wait()
        shutil.rmtree(self._tmp, ignore_errors=True)

    async def _restart(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        if self._reader:
            self._reader.cancel()
        await self.start()

    async def _send(self, cmd: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _command(self, cmd: dict, timeout: float = 15.0) -> dict:
        self._seq += 1
        cid = f"c{self._seq}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[cid] = fut
        try:
            await self._send({**cmd, "id": cid})
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(cid, None)

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                self._dispatch(rec)
        except (asyncio.CancelledError, ConnectionError, ValueError):
            return
        finally:
            if self._proc is proc:
                self._emit(("error", "o Pi encerrou"))
                self._idle.set()
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(LLMError("o Pi encerrou"))

    def _emit(self, item: tuple) -> None:
        if self._sink is not None:
            self._sink.put_nowait(item)

    def _dispatch(self, rec: dict) -> None:
        kind = rec.get("type")
        if kind == "response":
            fut = self._pending.get(rec.get("id", ""))
            if fut and not fut.done():
                fut.set_result(rec)
        elif kind == "message_update":
            ev = rec.get("assistantMessageEvent") or {}
            if ev.get("type") == "text_delta" and ev.get("delta"):
                self._emit(("delta", ev["delta"]))
        elif kind == "message_end":
            msg = rec.get("message") or {}
            if msg.get("role") == "assistant":
                if msg.get("stopReason") == "error":
                    self._emit(("error", str(msg.get("errorMessage") or "erro do provedor")))
                elif msg.get("stopReason") != "aborted":
                    self._emit(("end", None))
        elif kind == "agent_start":
            self._idle.clear()
        elif kind == "agent_settled":
            self._idle.set()

    # --- uso ----------------------------------------------------------
    async def new_session(self) -> None:
        async with self._lock:
            await self._ready()
            resp = await self._command({"type": "new_session"})
            if not resp.get("success"):
                raise LLMError(str(resp.get("error")))

    async def _ready(self) -> None:
        if not self._proc or self._proc.returncode is not None:
            await self.start()
        try:
            await asyncio.wait_for(self._idle.wait(), 5)
        except asyncio.TimeoutError:
            await self._restart()

    async def _abort(self) -> None:
        try:
            await self._command({"type": "abort"}, timeout=5)
            await asyncio.wait_for(self._idle.wait(), 5)
        except (asyncio.TimeoutError, LLMError, OSError):
            await self._restart()

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with self._lock:
            await self._ready()
            sink: asyncio.Queue = asyncio.Queue()
            self._sink = sink
            finished = False
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            self.first_token_s = None
            try:
                resp = await self._command({"type": "prompt", "message": prompt})
                if not resp.get("success"):
                    raise LLMError(str(resp.get("error")))
                while True:
                    try:
                        kind, value = await asyncio.wait_for(sink.get(), self.timeout)
                    except asyncio.TimeoutError as exc:
                        raise LLMError("o Pi não respondeu a tempo") from exc
                    if kind == "delta":
                        if self.first_token_s is None:
                            self.first_token_s = loop.time() - t0
                        yield value
                    elif kind == "end":
                        finished = True
                        return
                    else:
                        finished = True
                        raise LLMError(value)
            finally:
                self._sink = None
                if not finished:
                    await self._abort()


class FakeLLM(LLM):
    """Substituto sem rede: devolve respostas roteirizadas em pedaços pequenos, com atraso opcional."""

    def __init__(self, replies: list[str] | Callable[[str], str] | None = None, first_delay: float = 0.0,
                 chunk_delay: float = 0.0, chunk_size: int = 7):
        self.replies = replies if replies is not None else ["[smug] Tá, fingi que entendi. Agora me conta o resto."]
        self.first_delay, self.chunk_delay, self.chunk_size = first_delay, chunk_delay, chunk_size
        self.prompts: list[str] = []
        self.sessions = 1
        self.aborted = 0
        self.completions: list[str] = []
        self.completion_reply = '{"facts": [], "jokes": [], "summary": ""}'

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def new_session(self) -> None:
        self.sessions += 1

    async def complete(self, prompt: str) -> str:
        self.completions.append(prompt)
        return self.completion_reply

    def _next(self, prompt: str) -> str:
        if callable(self.replies):
            return self.replies(prompt)
        idx = min(len(self.prompts) - 1, len(self.replies) - 1)
        return self.replies[idx]

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        text = self._next(prompt)
        done = False
        try:
            await asyncio.sleep(self.first_delay)
            for i in range(0, len(text), self.chunk_size):
                yield text[i:i + self.chunk_size]
                await asyncio.sleep(self.chunk_delay)
            done = True
        finally:
            if not done:
                self.aborted += 1
