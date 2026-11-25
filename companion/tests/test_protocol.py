import json

from companion import protocol


def test_encode_is_one_utf8_json_line():
    raw = protocol.encode(protocol.say("u1", "Você de novo?", "smug"))
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert json.loads(raw) == {"t": "say", "id": "u1", "text": "Você de novo?", "emotion": "smug"}
    assert "Você".encode() in raw  # sem escapes \u00e9


def test_unknown_emotion_falls_back_to_neutral():
    assert protocol.say("u1", "oi", "furious")["emotion"] == "neutral"
    assert protocol.emotion("furious")["name"] == "neutral"


def test_mouth_weights_are_clamped_to_unit_range():
    m = protocol.mouth(aa=1.7, ih=-0.2, oh=0.5)
    assert (m["aa"], m["ih"], m["oh"]) == (1.0, 0.0, 0.5)
    assert set(m) == {"t", "aa", "ih", "ou", "ee", "oh"}


def test_decode_discards_invalid_lines():
    assert protocol.decode(b"not json\n") is None
    assert protocol.decode(b"[1,2]\n") is None
    assert protocol.decode(b'{"no_type":1}\n') is None
    assert protocol.decode(b"\xff\xfe\n") is None
    assert protocol.decode(b'{"t":"poke","kind":"pat","extra":1}\n') == {"t": "poke", "kind": "pat", "extra": 1}
