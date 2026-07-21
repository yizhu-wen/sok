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
import sys
import traceback


def serve(handlers: dict) -> None:
    """Serve requests until stdin closes or a 'shutdown' command arrives.

    handlers: {"embed": fn, "decode": fn, "ping": fn, ...}; each fn takes the
    decoded request dict and returns a response dict (without the 'ok' key).
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                return
            if cmd not in handlers:
                raise ValueError(f"unknown cmd {cmd!r}")
            resp = dict(handlers[cmd](req))
            resp["ok"] = True
        except Exception as e:  # noqa: BLE001 — worker must never die on a bad request
            traceback.print_exc()
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(resp), flush=True)
