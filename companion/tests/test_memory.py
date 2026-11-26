import json

from companion.brain.memory import Memory, parse_summary_reply


def make(tmp_path, **kw):
    t = [1000.0]
    def clock():
        t[0] += 1
        return t[0]
    return Memory(tmp_path / "mem.json", clock=clock, **kw)


def test_window_is_rolling_and_persisted(tmp_path):
    m = make(tmp_path, window=4)
    for i in range(6):
        m.add("user" if i % 2 == 0 else "companion", f"msg {i}")
    assert [x.text for x in m.data.recent] == ["msg 2", "msg 3", "msg 4", "msg 5"]
    again = Memory(tmp_path / "mem.json", window=4)
    assert [x.text for x in again.data.recent] == ["msg 2", "msg 3", "msg 4", "msg 5"]


def test_summary_trigger_counts_only_new_messages(tmp_path):
    m = make(tmp_path, window=30, summarize_after=4)
    for i in range(3):
        m.add("user", f"m{i}")
    assert not m.needs_summary()
    m.add("companion", "m3")
    assert m.needs_summary()


async def test_summarize_merges_facts_and_resets_trigger(tmp_path):
    m = make(tmp_path, summarize_after=2)
    m.data.facts = ["gosta de café"]
    m.add("user", "meu gato se chama Bolinha")
    m.add("companion", "Bolinha é nome de bala também")
    prompts = []

    async def complete(prompt):
        prompts.append(prompt)
        return 'claro:\n```json\n{"facts": ["gosta de café", "gato chamado Bolinha", "Gosta de CAFÉ"], "jokes": ["nome de bala"], "summary": "falaram do gato"}\n```'

    assert await m.summarize(complete)
    assert m.data.facts == ["gosta de café", "gato chamado Bolinha"]
    assert m.data.jokes == ["nome de bala"] and m.data.summary == "falaram do gato"
    assert not m.needs_summary()
    assert "Bolinha" in prompts[0] and "gosta de café" in prompts[0]
    assert json.loads((tmp_path / "mem.json").read_text())["facts"][1] == "gato chamado Bolinha"


async def test_failed_summary_keeps_messages_for_next_try(tmp_path):
    m = make(tmp_path, summarize_after=2)
    m.add("user", "a")
    m.add("user", "b")

    async def broken(prompt):
        return "não sei fazer json"

    async def boom(prompt):
        raise RuntimeError("rede")

    assert not await m.summarize(broken)
    assert not await m.summarize(boom)
    assert m.needs_summary()


def test_parse_summary_reply_tolerates_noise():
    assert parse_summary_reply('texto {"facts": []} fim') == {"facts": []}
    assert parse_summary_reply("nada") is None
    assert parse_summary_reply("{quebrado") is None


def test_prompt_block_has_facts_jokes_and_recent(tmp_path):
    m = make(tmp_path)
    m.data.facts, m.data.jokes, m.data.summary = ["tem um gato"], ["o Alt+Tab"], "começaram a se conhecer"
    m.add("user", "oi")
    m.add("companion", "e aí", interrupted=True)
    block = m.prompt_block("ele", "você")
    for piece in ("tem um gato", "o Alt+Tab", "começaram a se conhecer", "ele: oi", "você: e aí [interrompida]", "Última conversa"):
        assert piece in block


def test_corrupt_file_starts_empty(tmp_path):
    (tmp_path / "mem.json").write_text("{{{")
    assert make(tmp_path).data.recent == []


async def test_messages_arriving_during_summary_stay_fresh_when_window_rolls(tmp_path):
    m = make(tmp_path, window=3, summarize_after=2)
    m.add("user", "meu gato chama Bolinha")
    m.add("companion", "Tá anotado no meu coração de silício.")
    m.add("user", "prefiro café sem açúcar")

    async def complete(prompt):
        assert "Bolinha" in prompt and "café" in prompt
        m.add("user", "meu cachorro chama Pistache")
        m.add("companion", "Sua casa virou uma sorveteria.")
        return '{"facts": ["gato chamado Bolinha", "prefere café sem açúcar"]}'

    assert await m.summarize(complete)
    reloaded = make(tmp_path, window=3, summarize_after=2)
    assert reloaded.needs_summary()
    assert reloaded.data.unsummarized == 2
