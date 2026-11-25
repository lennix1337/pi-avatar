#!/usr/bin/env python3
"""Fake companion for the pet: a scripted ~20 s conversation speaking companion/PROTOCOL.md.

  python fake_companion.py serve [--port 47800] [--loop]   # TCP server the pet connects to
  python fake_companion.py emit talk-20s.jsonl [--lead 2]  # timestamped stream for --companion-replay

Both modes play the same script. The mouth is streamed at 30 Hz with a syllable-like envelope
and a cycling vowel, like a real lip-sync feed. Lines the pet sends (hello, poke) are printed.
Standard library only.
"""
import argparse
import json
import math
import socket
import sys
import threading
import time

MOUTH_HZ = 30.0
VOWELS = ["aa", "ih", "ou", "ee", "oh"]


def mouth_frames(t0, duration, seed):
    """Mouth messages at 30 Hz over [t0, t0 + duration): syllables of ~0.17 s, a vowel each."""
    out = []
    n = int(duration * MOUTH_HZ)
    for i in range(n):
        t = i / MOUTH_HZ
        syllable = int(t / 0.17)
        phase = (t % 0.17) / 0.17
        level = math.sin(math.pi * phase) ** 0.7 * (0.55 + 0.4 * ((syllable * 7 + seed) % 5) / 4)
        vowel = VOWELS[(syllable * 3 + seed) % 5]
        msg = {"t": "mouth", vowel: round(level, 3)}
        if (syllable + seed) % 4 == 0:
            msg[VOWELS[(syllable + 1) % 5]] = round(level * 0.3, 3)
        out.append((t0 + t, msg))
    return out


def script():
    """The conversation as sorted (seconds, message) pairs, 20 s long."""
    ev = []

    def at(when, **m):
        ev.append((when, m))

    at(0.0, t="state", listening=True)
    at(2.0, t="gesture", name="nod")
    at(3.0, t="state", listening=False, thinking=True)
    # 1: friendly greeting.
    at(4.5, t="say", id="u1", text="Oi! Eu sou a Pi. Agora consigo falar com você de verdade.", emotion="happy")
    at(4.5, t="state", thinking=False, speaking=True)
    at(5.0, t="gesture", name="wave")
    ev += mouth_frames(4.7, 3.6, 1)
    at(8.4, t="end", id="u1")
    at(8.4, t="state", speaking=False)
    # 2: smug, longer text that has to scroll.
    at(9.5, t="say", id="u2", text="Eu avisei que ia dar certo. Os testes passaram de primeira, o build ficou verde e ninguém precisou abrir o depurador. Mais um para a conta.", emotion="smug")
    at(9.5, t="state", speaking=True)
    at(11.0, t="gesture", name="tilt")
    ev += mouth_frames(9.7, 6.0, 2)
    at(15.8, t="end", id="u2")
    at(15.8, t="state", speaking=False)
    # 3: embarrassed, short.
    at(16.8, t="say", id="u3", text="Ah... não precisa elogiar tanto.", emotion="embarrassed")
    at(16.8, t="state", speaking=True)
    at(17.4, t="gesture", name="shake")
    ev += mouth_frames(17.0, 2.2, 3)
    at(19.3, t="end", id="u3")
    at(19.3, t="state", speaking=False)
    at(20.0, t="emotion", name="neutral", intensity=0.0)
    ev.sort(key=lambda p: p[0])
    return ev


def emit(path, lead):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t, m in script():
            f.write(json.dumps({"at": round(t + lead, 3), **m}, separators=(",", ":")) + "\n")


def reader(conn):
    buf = b""
    while True:
        try:
            chunk = conn.recv(4096)
        except OSError:
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            print("pet ->", line.decode("utf-8", "replace"), flush=True)


def serve(port, loop):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print(f"fake companion on 127.0.0.1:{port}", flush=True)
    while True:
        conn, _ = srv.accept()
        print("pet connected", flush=True)
        threading.Thread(target=reader, args=(conn,), daemon=True).start()
        try:
            while True:
                start = time.monotonic()
                for t, m in script():
                    wait = start + t - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    conn.sendall((json.dumps(m, separators=(",", ":")) + "\n").encode())
                if not loop:
                    time.sleep(2.0)
                    break
        except OSError:
            print("pet disconnected", flush=True)
            continue
        conn.close()
        if not loop:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=47800)
    s.add_argument("--loop", action="store_true")
    e = sub.add_parser("emit")
    e.add_argument("path")
    e.add_argument("--lead", type=float, default=0.0, help="seconds of quiet before the script starts")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.port, a.loop)
    else:
        emit(a.path, a.lead)


if __name__ == "__main__":
    sys.exit(main())
