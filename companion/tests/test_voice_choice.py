import importlib.util

import pytest

from companion.__main__ import create_voice
from companion.config import Config, load_config
from companion.fakevoice import FakeVoice
from companion.voiceapi import Utterance, VoiceIO


def test_text_flag_or_config_picks_the_typed_voice():
    assert isinstance(create_voice(Config(), text=True), FakeVoice)
    assert isinstance(create_voice(Config(text_mode=True)), FakeVoice)


def test_voice_section_of_companion_json_is_loaded(tmp_path):
    p = tmp_path / "companion.json"
    p.write_text('{"text_mode": true, "voice": {"tts": {"backend": "kokoro"}, "output": {"device": 3}}}', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.text_mode and cfg.voice["tts"]["backend"] == "kokoro"


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="extra `voice` não instalado")
def test_real_voice_is_the_default_and_shares_the_brain_utterance():
    import companion.voice as v

    voice = create_voice(Config(voice={"tts": {"backend": "fake"}}))
    assert isinstance(voice, v.Voice) and isinstance(voice, VoiceIO)
    assert v.Utterance is Utterance
    assert voice.cfg["tts"]["backend"] == "fake"
