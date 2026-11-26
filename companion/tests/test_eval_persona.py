import subprocess
import sys
from pathlib import Path


def test_offline_persona_eval_runs_all_turns_and_checks():
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_persona.py"
    run = subprocess.run([sys.executable, "-X", "utf8", str(script)], capture_output=True, text=True,
                         encoding="utf-8", timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    assert sum(line[:2].isdigit() for line in run.stdout.splitlines()) == 15
    assert run.stdout.count("PASS:") == 4
    assert "reinício" in run.stdout and "Farofa" in run.stdout and "sem açúcar" in run.stdout
