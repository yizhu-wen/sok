"""
Minimal JSON-lines request/response loop shared by the per-method demo
workers.  Each worker runs inside its own virtualenv (the methods have
mutually incompatible dependencies) and talks to the main Gradio process
over stdin/stdout:

    request:   {"cmd": "<name>", ...}\n
    response:  {"ok": true, ...}\n   or   {"ok": false, "error": "..."}\n

Audio is exchanged through float32 WAV files rather than inline JSON.
Stderr is left free for library logging and tracebacks.
"""
from __future__ import annotations

import json
import os
import sys
import traceback


_proto = None


def hijack_stdout():
    """Reserve the stdout pipe for protocol replies only.

    Keeps a private duplicate of the original stdout and points fd 1 (and
    sys.stdout) at stderr, so stray print()s in vendored model code — e.g.
    hifigan's "Removing weight norm..." at load time and Timbre's prints
    during inference — cannot corrupt the JSON-lines channel.  Workers must
    call this immediately after importing this module, BEFORE importing or
    building any model code.
    """
    global _proto
    if _proto is None:
        _proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
    return _proto


def serve(handlers: dict) -> None:
    """Serve requests until stdin closes or a 'shutdown' command arrives.

    handlers: {"embed": fn, "decode": fn, "ping": fn, ...}; each fn takes the
    decoded request dict and returns a response dict (without the 'ok' key).
    """
    proto = hijack_stdout()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "shutdown":
                proto.write(json.dumps({"ok": True}) + "\n")
                proto.flush()
                return
            if cmd not in handlers:
                raise ValueError(f"unknown cmd {cmd!r}")
            resp = dict(handlers[cmd](req))
            resp["ok"] = True
        except Exception as e:  # noqa: BLE001 — worker must never die on a bad request
            traceback.print_exc()
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        proto.write(json.dumps(resp) + "\n")
        proto.flush()
