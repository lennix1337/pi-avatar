import asyncio
import sys
from pathlib import Path

import pytest

from companion.brain.llm import LLMError, PiRpcLLM

FAKE = str(Path(__file__).with_name("fake_pi.py"))


def make(tmp_path, mode="ok"):
    llm = PiRpcLLM([sys.executable, FAKE, f"--mode-test={mode}"], "x/y", "persona", timeout=5, workdir=tmp_path / "w")
    return llm


async def collect(llm, prompt):
    return "".join([c async for c in llm.stream(prompt)])


async def test_streams_text_deltas_and_reuses_the_process(tmp_path):
    llm = make(tmp_path)
    try:
        assert await collect(llm, "ele: oi") == "[smug] Eco 1: ele: oi. Fim."
        pid = llm._proc.pid
        assert (await collect(llm, "ele: de novo")).endswith("ele: de novo. Fim.")
        assert llm._proc.pid == pid
        assert llm.first_token_s is not None
    finally:
        await llm.close()


async def test_cli_args_have_no_tools_no_extensions_and_a_persona_file(tmp_path):
    llm = make(tmp_path)
    argv = llm.argv()
    for flag in ("--no-tools", "--system-prompt", "--no-extensions", "--no-session", "--mode", "rpc", "--thinking", "off"):
        assert flag in argv
    assert (tmp_path / "w" / "persona.md").read_text() == "persona"


async def test_provider_error_raises_llmerror(tmp_path):
    llm = make(tmp_path, "error")
    try:
        with pytest.raises(LLMError, match="529"):
            await collect(llm, "ele: oi")
    finally:
        await llm.close()


async def test_closing_the_stream_aborts_and_the_next_prompt_still_works(tmp_path):
    llm = make(tmp_path, "slow")
    try:
        gen = llm.stream("ele: um texto comprido para dar tempo")
        assert await gen.__anext__()
        await gen.aclose()
        assert llm._idle.is_set()
        assert (await collect(llm, "ele: depois")).endswith("depois. Fim.")
    finally:
        await llm.close()


async def test_cancelling_the_consumer_task_aborts_generation(tmp_path):
    llm = make(tmp_path, "slow")
    try:
        got = []

        async def consume():
            async for c in llm.stream("ele: texto bem comprido para cancelar no meio"):
                got.append(c)

        task = asyncio.create_task(consume())
        while not got:
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert llm._idle.is_set()
    finally:
        await llm.close()


async def test_complete_starts_a_fresh_session_each_time(tmp_path):
    llm = make(tmp_path)
    try:
        assert "Eco 2" in await llm.complete("a")
        assert "Eco 3" in await llm.complete("b")
    finally:
        await llm.close()


async def test_fake_llm_exposes_only_current_session_prompts_to_replay():
    from companion.brain.llm import FakeLLM

    llm = FakeLLM(lambda _: str(len(llm.session_prompts)))
    assert "".join([s async for s in llm.stream("um")]) == "1"
    assert "".join([s async for s in llm.stream("dois")]) == "2"
    await llm.new_session()
    assert "".join([s async for s in llm.stream("três")]) == "1"
    assert llm.session_prompts == ["três"]
    assert llm.prompts == ["um", "dois", "três"]
