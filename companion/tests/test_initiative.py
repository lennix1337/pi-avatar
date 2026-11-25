import random

from companion.brain.initiative import Harness, Initiative, read_harness
from companion.config import InitiativeConfig

DAY = 100
NOON = 14


def make(**kw):
    cfg = InitiativeConfig(silence_s=100.0, jitter=0.0, **kw)
    return Initiative(cfg, random.Random(1))


def test_first_check_is_a_greeting_then_silence_waits_the_interval():
    ini = make()
    assert ini.check(0, NOON, DAY).kind == "greeting"
    assert ini.check(50, NOON, DAY) is None
    trig = ini.check(101, NOON, DAY)
    assert trig and trig.kind == "silence"


def test_each_unanswered_initiative_doubles_the_wait():
    ini = make()
    ini.check(0, NOON, DAY)
    ini.check(101, NOON, DAY)
    ini.on_companion_spoke(110, initiated=True)          # sem resposta: próximo em 200 s
    assert ini.check(110 + 150, NOON, DAY) is None
    assert ini.check(110 + 201, NOON, DAY).kind == "silence"
    ini.on_companion_spoke(330, initiated=True)          # 400 s
    assert ini.check(330 + 399, NOON, DAY) is None
    assert ini.check(330 + 401, NOON, DAY).kind == "silence"


def test_stops_after_max_unanswered_until_he_talks():
    ini = make(max_unanswered=2)
    ini.check(0, NOON, DAY)
    ini.on_companion_spoke(10, True)
    ini.on_companion_spoke(20, True)
    assert ini.check(100000, NOON, DAY) is None
    ini.on_user_activity(100001)
    assert ini.unanswered == 0
    assert ini.check(100001 + 101, NOON, DAY).kind == "silence"


def test_user_reply_resets_backoff():
    ini = make()
    ini.check(0, NOON, DAY)
    ini.on_companion_spoke(10, True)
    ini.on_user_activity(20)
    assert ini.check(20 + 101, NOON, DAY).kind == "silence"


def test_greeting_and_answers_to_him_do_not_count_as_unanswered():
    ini = make()
    ini.check(0, NOON, DAY)
    ini.on_companion_spoke(5, initiated=False)
    assert ini.unanswered == 0


def test_quiet_at_night_unless_he_is_talking():
    ini = make()
    ini.check(0, 3, DAY)                                 # cumprimento é permitido na partida
    assert ini.check(5000, 3, DAY) is None               # 3h da manhã, ele está quieto
    ini.on_user_activity(5000)
    assert ini.check(5000 + 101, 3, DAY).kind in ("late", "silence")


def test_late_night_remark_fires_once_per_night_while_he_is_active():
    ini = make()
    ini.check(0, 2, DAY)
    ini.on_user_activity(10)
    first = ini.check(20, 2, DAY)
    assert first and first.kind == "late"
    ini.on_user_activity(30)
    assert ini.check(40, 2, DAY) is None
    ini.on_user_activity(50)
    assert ini.check(60, 2, DAY + 1).kind == "late"


def test_harness_events_fire_on_rising_edge_with_cooldown():
    ini = Initiative(InitiativeConfig(silence_s=1e9, jitter=0.0), random.Random(1))
    ini.check(0, NOON, DAY)
    assert ini.check(1, NOON, DAY, Harness()) is None
    assert ini.check(2, NOON, DAY, Harness(ci_red=True)).kind == "ci_red"
    assert ini.check(3, NOON, DAY, Harness(ci_red=True)) is None             # ainda vermelho, sem repetir
    assert ini.check(4, NOON, DAY, Harness()) is None
    assert ini.check(5, NOON, DAY, Harness(ci_red=True)) is None             # cooldown de 600 s
    assert ini.check(700, NOON, DAY, Harness()) is None
    assert ini.check(701, NOON, DAY, Harness(ci_red=True)).kind == "ci_red"


def test_harness_stays_silent_at_night_when_he_is_away():
    ini = make()
    ini.check(0, 3, DAY)
    assert ini.check(1, 3, DAY, Harness()) is None
    assert ini.check(2, 3, DAY, Harness(merged=True)) is None


def test_disabled_never_speaks():
    ini = Initiative(InitiativeConfig(enabled=False))
    assert ini.check(0, NOON, DAY) is None and ini.check(99999, NOON, DAY) is None


def test_poke_replies_have_a_cooldown():
    ini = make()
    assert ini.poke_reply_allowed(0)
    assert not ini.poke_reply_allowed(5)
    assert ini.poke_reply_allowed(25)


def test_jitter_stays_within_bounds():
    ini = Initiative(InitiativeConfig(silence_s=100.0, jitter=0.3), random.Random(7))
    vals = [ini.interval() for _ in range(200)]
    assert all(70 <= v <= 130 for v in vals) and len(set(vals)) > 50


def test_read_harness_matches_the_pet_json(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"ciRed": true, "workers": 3, "waitingOperator": true, "idleMinutes": 4.5}')
    assert read_harness(p) == Harness(ci_red=True, merged=False, workers=3, waiting_operator=True, idle_minutes=4.5)
    p.write_text("lixo")
    assert read_harness(p) is None and read_harness(tmp_path / "nope.json") is None and read_harness(None) is None
