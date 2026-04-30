"""
Persistent JSON-lines worker for speaker encoder cosine similarity (SECS).

Protocol:
  request:  {"cmd": "compare_many", "ref_path": "...", "deg_paths": ["...", ...]}
  response: {"ok": true, "scores": [0.99, 0.87, ...]}

The worker keeps a single Resemblyzer encoder alive to amortize model load cost.
It runs best in a speech-oriented env that already has torch/librosa/soundfile.
"""

import json
import sys
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


def _emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _score_pair(encoder, ref_path, deg_path):
    ref_wav = preprocess_wav(Path(ref_path))
    deg_wav = preprocess_wav(Path(deg_path))
    ref_embed = encoder.embed_utterance(ref_wav)
    deg_embed = encoder.embed_utterance(deg_wav)
    return float(np.dot(ref_embed, deg_embed))


def main():
    encoder = VoiceEncoder(verbose=False, device="cpu")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "shutdown":
                _emit({"ok": True})
                return
            if cmd != "compare_many":
                raise ValueError(f"unknown command: {cmd!r}")

            ref_path = req["ref_path"]
            deg_paths = req["deg_paths"]
            scores = []
            for deg_path in deg_paths:
                try:
                    scores.append(_score_pair(encoder, ref_path, deg_path))
                except Exception:
                    scores.append(None)
            _emit({"ok": True, "scores": scores})
        except Exception as e:
            _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
