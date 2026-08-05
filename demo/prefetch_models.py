"""
Best-effort model checkpoint prefetch, run at Docker build time so the demo
starts without needing network access.  Each model is attempted independently;
failures are reported but never fail the build (the demo lazily retries at
runtime).
"""
from __future__ import annotations

import sys
import traceback


def _try(name, fn):
    try:
        fn()
        print(f"[prefetch] {name}: OK", flush=True)
        return True
    except Exception as e:
        print(f"[prefetch] {name}: FAILED ({type(e).__name__}: {e})", flush=True)
        traceback.print_exc()
        return False


def prefetch_audioseal():
    from audioseal import AudioSeal
    AudioSeal.load_generator("audioseal_wm_16bits")
    AudioSeal.load_detector("audioseal_detector_16bits")


def prefetch_wavmark():
    import wavmark
    wavmark.load_model()


def prefetch_silentcipher():
    import silentcipher
    silentcipher.get_model(model_type="44.1k", device="cpu")


def main():
    results = [
        _try("audioseal", prefetch_audioseal),
        _try("wavmark", prefetch_wavmark),
        _try("silentcipher", prefetch_silentcipher),
    ]
    print(f"[prefetch] {sum(results)}/{len(results)} model sets cached", flush=True)
    sys.exit(0)  # never fail the image build


if __name__ == "__main__":
    main()
