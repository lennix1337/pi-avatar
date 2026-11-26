import json
from pathlib import Path

from companion.brain.tags import parse_segment
from companion.protocol import EMOTIONS


def test_every_brain_emotion_tag_has_an_avatar_profile():
    library = json.loads((Path(__file__).resolve().parents[2] / "poses.json").read_text(encoding="utf-8"))
    assert set(library["emotions"]) == set(EMOTIONS)
    for name in EMOTIONS:
        assert parse_segment(f"[{name}] Oi").emotion == name


def test_emotion_and_gesture_are_extracted_and_stripped():
    seg = parse_segment("[smug] Ah, então agora você acredita. [gesture:tilt]")
    assert seg.text == "Ah, então agora você acredita."
    assert seg.emotion == "smug"
    assert seg.gestures == ["tilt"]


def test_bare_gesture_name_and_case_and_spaces():
    seg = parse_segment("[ Happy ] Opa! [laugh] Bora.")
    assert seg.emotion == "happy" and seg.gestures == ["laugh"]
    assert seg.text == "Opa! Bora."


def test_first_emotion_wins_and_unknown_tags_are_not_spoken():
    seg = parse_segment("[sad] Hm. [excited] Ué. [risos] Tá.")
    assert seg.emotion == "sad"
    assert seg.text == "Hm. Ué. Tá."


def test_invalid_gesture_is_dropped_without_leaking_into_text():
    seg = parse_segment("[gesture:moonwalk] Oi.")
    assert seg.gestures == [] and seg.text == "Oi."


def test_text_without_tags_is_untouched():
    seg = parse_segment("Tá, né?")
    assert (seg.text, seg.emotion, seg.gestures) == ("Tá, né?", None, [])


def test_only_tags_gives_empty_text():
    assert parse_segment("[gesture:wave]").text == ""
