"""Imita `pi --mode rpc` o suficiente para testar o cliente: prompt → deltas → message_end → agent_settled."""
import json
import queue
import sys
import threading
import time

mode = "ok"
for a in sys.argv:
    if a.startswith("--mode-test="):
        mode = a.split("=", 1)[1]

commands: queue.Queue = queue.Queue()
aborted = threading.Event()


def out(rec):
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()


def read_stdin():
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd["type"] == "abort":
            aborted.set()
        commands.put(cmd)
    commands.put(None)


threading.Thread(target=read_stdin, daemon=True).start()
sessions = 1
while (cmd := commands.get()) is not None:
    kind, cid = cmd["type"], cmd.get("id")
    if kind == "new_session":
        sessions += 1
        out({"id": cid, "type": "response", "command": kind, "success": True, "data": {"cancelled": False}})
    elif kind == "abort":
        out({"id": cid, "type": "response", "command": kind, "success": True})
    elif kind == "prompt":
        aborted.clear()
        out({"id": cid, "type": "response", "command": kind, "success": True, "data": {"disposition": "started"}})
        out({"type": "agent_start"})
        if mode == "error":
            out({"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": "529 overloaded"}})
            out({"type": "agent_settled", "aborted": False})
            continue
        text = f"[smug] Eco {sessions}: " + cmd["message"].splitlines()[-1] + ". Fim."
        stopped = False
        for i in range(0, len(text), 6):
            if aborted.is_set():
                stopped = True
                break
            out({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": text[i:i + 6]}})
            time.sleep(0.2 if mode == "slow" else 0.0)
        out({"type": "message_end", "message": {"role": "assistant", "stopReason": "aborted" if stopped else "stop"}})
        out({"type": "agent_settled", "aborted": stopped})
