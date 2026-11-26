import re

from companion.brain.persona import load_persona
from companion.config import Config


def test_bundled_persona_examples_model_brief_spoken_replies():
    text = load_persona(Config(name="Pipoca"))
    assert "Pipoca" in text and "{name}" not in text
    examples = re.findall(r"^você: (.+)$", text, re.M)
    assert len(examples) >= 6
    for example in examples:
        spoken = re.sub(r"\[[^\]]+\]", "", example)
        assert len(re.findall(r"[.!?…]+(?=\s|[\"”]|$)", spoken)) <= 2, example
