from companion.brain.chunker import SentenceChunker


def feed_all(text, size):
    c = SentenceChunker()
    out = []
    for i in range(0, len(text), size):
        out += c.feed(text[i:i + size])
    return out + c.flush()


def test_sentences_stream_out_as_soon_as_they_end():
    c = SentenceChunker()
    assert c.feed("[smug] Ah, então é isso? ") == ["[smug] Ah, então é isso?"]
    assert c.feed("Eu falei, né") == []
    assert c.feed(". Agora") == ["Eu falei, né."]
    assert c.flush() == ["Agora"]


def test_result_is_the_same_for_any_delta_size():
    text = "[happy] Oi, tudo bem? [gesture:wave] Eu tava pensando em você... Brincadeira! Ou não."
    expected = feed_all(text, len(text))
    for size in (1, 2, 3, 5, 8, 13):
        assert feed_all(text, size) == expected


def test_tag_is_never_split_across_sentences():
    out = feed_all("Tá bom demais. [gesture:tilt] [teasing] Sei.", 3)
    assert out == ["Tá bom demais.", "[gesture:tilt] [teasing] Sei."]


def test_short_fragment_waits_for_the_next_sentence():
    assert feed_all("Hm. Então, deixa eu pensar um pouco nisso.", 4) == ["Hm. Então, deixa eu pensar um pouco nisso."]


def test_short_question_or_exclamation_is_spoken_right_away():
    c = SentenceChunker()
    assert c.feed("Ué? Pois é") == ["Ué?"]


def test_decimals_and_ellipsis_do_not_cut_mid_word():
    out = feed_all("Custa 3.5 reais mesmo... Que absurdo isso, hein.", 2)
    assert out == ["Custa 3.5 reais mesmo...", "Que absurdo isso, hein."]


def test_long_run_without_punctuation_is_cut_at_a_comma():
    text = "palavra " * 12 + "e mais, " + "coisa " * 20
    c = SentenceChunker()
    first = c.feed(text)
    assert first and len(first[0]) <= 160 and first[0].endswith(",")


def test_flush_ignores_tag_only_leftovers():
    c = SentenceChunker()
    c.feed("Fim de papo. [gesture:wave]")
    assert c.flush() == []
