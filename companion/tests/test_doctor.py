from companion.config import Config
from companion.doctor import render, run_checks


def _mods(missing=()):
    def importer(name):
        if name in missing:
            raise ImportError(name)
        return object()

    return importer


DEVS = [
    {"index": 0, "name": "Mic USB", "max_input_channels": 1, "max_output_channels": 0},
    {"index": 1, "name": "Caixa", "max_input_channels": 0, "max_output_channels": 2},
]


def run(cfg=None, **kw):
    args = dict(which=lambda c: "/bin/pi", port_free=lambda h, p: True, devices=lambda: DEVS, importer=_mods())
    args.update(kw)
    return {c.name: c for c in run_checks(cfg or Config(), **args)}


def test_all_green():
    r = run()
    assert all(c.ok for c in r.values()) and render(list(r.values())).endswith("tudo certo")


def test_missing_pi_and_busy_port_have_fixes():
    r = run(which=lambda c: None, port_free=lambda h, p: False)
    assert not r["pi"].ok and "pi_command" in r["pi"].fix
    assert not r["porta do pet"].ok and "--port" in r["porta do pet"].fix


def test_missing_voice_extra_stops_before_devices():
    def boom():
        raise AssertionError("não deve listar dispositivos")

    r = run(importer=_mods({"sounddevice"}), devices=boom)
    assert not r["extra de voz"].ok and "sounddevice" in r["extra de voz"].detail and "áudio" not in r


def test_unknown_configured_device_fails():
    r = run(Config(voice={"listen": {"input_device": 9}, "output": {"device": "Caixa"}}))
    assert not r["microfone"].ok and r["caixa de som"].ok


def test_no_output_device_and_portaudio_error():
    r = run(devices=lambda: DEVS[:1])
    assert r["microfone"].ok and not r["caixa de som"].ok

    def err():
        raise OSError("sem PortAudio")

    assert not run(devices=err)["áudio"].ok


def test_text_mode_skips_audio():
    assert set(run(Config(text_mode=True))) == {"pi", "porta do pet"}
