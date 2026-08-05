"""
Interactive Gradio demo for the audio-watermarking robustness benchmark.

Workflow
  1) Upload a single audio file.
  2) Select one, several, or all watermark methods; optionally edit the
     payload bits per method (bit-length constraints are enforced).
  3) Embed → listen to each watermarked output + its spectrogram.
  4) Select digital-level distortions (each selected type applies all of its
     settings by default; individual settings can be toggled).
  5) Evaluate → for every distorted watermarked audio the app reports
     bit recovery rate, SI-SNR, ViSQOL, and the decoded payload bits
     ("None" when a method cannot decode anything).

Run:  python demo/app.py       (inside the Docker image this is the CMD)
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

import eval_utils as eu  # noqa: E402
import methods as wm  # noqa: E402

import gradio as gr  # noqa: E402
import librosa  # noqa: E402

TMP_ROOT = Path(os.environ.get("SOK_DEMO_TMPDIR",
                               os.path.join(tempfile.gettempdir(), "sok_demo")))
TMP_ROOT.mkdir(parents=True, exist_ok=True)

REGISTRY = wm.REGISTRY
METHOD_NAMES = [m.name for m in REGISTRY]
AVAILABLE_NAMES = [m.name for m in REGISTRY if m.unavailable_reason is None]

DIST_KEYS = list(eu.DIGITAL_DISTORTIONS)
# Checkbox labels carry the number of attack settings each type contributes,
# e.g. "Pitch shift (cents) · 10 settings".  CHOICE_TO_KEY maps back.
DIST_CHOICE_LABELS = {
    k: f"{eu.DISTORTION_LABELS[k]} · {len(v)} settings"
    for k, v in eu.DIGITAL_DISTORTIONS.items()
}
CHOICE_TO_KEY = {v: k for k, v in DIST_CHOICE_LABELS.items()}
DIST_LABELS = list(DIST_CHOICE_LABELS.values())
TOTAL_SETTINGS = sum(len(v) for v in eu.DIGITAL_DISTORTIONS.values())

MAX_EVAL_JOBS = int(os.environ.get("SOK_DEMO_MAX_JOBS", "1200"))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _new_session_dir() -> Path:
    return Path(tempfile.mkdtemp(dir=TMP_ROOT, prefix="session_"))


def _write_wav(path: Path, y: np.ndarray, sr: int) -> str:
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:  # keep the player from clipping loud classic-method output
        y = y / peak
    sf.write(str(path), y.astype(np.float32), sr, subtype="FLOAT")
    return str(path)


def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{digits}f}"


def _decoded_str(decoded) -> str:
    if decoded is None:
        return "None"
    return wm.bits_to_str(np.asarray(decoded).astype(int).ravel())


def _make_example_wav() -> str:
    """Synthesize a small clean example clip so the demo works out of the box."""
    path = TMP_ROOT / "example_input.wav"
    if path.exists():
        return str(path)
    sr, dur = 16000, 8.0
    t = np.arange(int(sr * dur)) / sr
    notes = [220.0, 277.18, 329.63, 440.0, 329.63, 277.18, 246.94, 220.0]
    seg = len(t) // len(notes)
    y = np.zeros_like(t, dtype=np.float64)
    for i, f0 in enumerate(notes):
        sl = slice(i * seg, (i + 1) * seg)
        tt = t[sl] - t[sl][0]
        env = np.minimum(1.0, tt * 30) * np.exp(-tt * 1.5)
        for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
            y[sl] += amp * env * np.sin(2 * np.pi * f0 * h * tt)
    y += 0.003 * np.random.RandomState(0).randn(len(y))
    y = 0.6 * y / np.max(np.abs(y))
    sf.write(str(path), y.astype(np.float32), sr)
    return str(path)


# ─── Step 1: embed ────────────────────────────────────────────────────────────

def do_embed(audio_path, selected_names, visqol_mode, *bit_texts,
             progress=gr.Progress()):
    n_out = len(REGISTRY)
    hidden = [gr.update(visible=False), gr.update(value=""),
              gr.update(value=None), gr.update(value=None)]

    def _all_hidden(status, state):
        return [status, state] + hidden * n_out

    if not audio_path:
        return _all_hidden("⚠️ Please upload an audio file first.", None)
    if not selected_names:
        return _all_hidden("⚠️ Select at least one watermark method.", None)

    stem = Path(audio_path).stem
    try:
        y, sr = eu.load_audio_clipped(audio_path, stem=stem)
    except Exception as e:
        return _all_hidden(f"⚠️ Could not read audio: {e}", None)

    session_dir = _new_session_dir()
    state = {"orig_y": y, "orig_sr": sr, "stem": stem,
             "visqol_mode": visqol_mode, "session_dir": str(session_dir),
             "wm": {}, "play_map": {}}
    # Rendered once here so the comparison view can show it without recomputing.
    try:
        state["orig_spec"] = eu.spectrogram_png(
            y, sr, f"Original — uploaded audio ({sr} Hz)",
            str(session_dir / "original.png"))
    except Exception:
        state["orig_spec"] = None

    per_method_updates = []
    ok, failed = [], []
    todo = [m for m in REGISTRY if m.name in selected_names]
    done = 0
    progress(0.0, desc="Loading models / embedding…")

    for m in REGISTRY:
        if m.name not in selected_names:
            per_method_updates += hidden
            continue
        header = f"### {m.name}  ·  {m.kind}"
        if m.unavailable_reason:
            per_method_updates += [
                gr.update(visible=True),
                gr.update(value=f"{header}\n\n⚠️ Unavailable: {m.unavailable_reason}"),
                gr.update(value=None), gr.update(value=None)]
            failed.append(m.name)
            done += 1
            continue
        try:
            bits = m.validate_bits(bit_texts[REGISTRY.index(m)])
            progress(done / max(len(todo), 1), desc=f"Embedding with {m.name}…")
            y_wm, sr_wm = m.embed(y, sr, bits)
            y_wm = np.asarray(y_wm, dtype=np.float32)

            ref = librosa.resample(y, orig_sr=sr, target_sr=sr_wm) \
                if sr != sr_wm else y
            ref_a, wm_a = eu.align_lengths(ref, y_wm)
            si = eu.compute_si_snr(ref_a, wm_a)

            clean_decoded = m.decode(y_wm, sr_wm, len(bits))
            clean_rec = eu.bit_recovery(bits, clean_decoded)

            wav_path = _write_wav(session_dir / f"{m.key}_watermarked.wav",
                                  y_wm, sr_wm)
            png_path = eu.spectrogram_png(
                y_wm, sr_wm, f"{m.name} — watermarked ({sr_wm} Hz)",
                str(session_dir / f"{m.key}_watermarked.png"))

            md = (f"{header}\n\n"
                  f"- payload ({len(bits)} bits): `{wm.bits_to_str(bits)}`\n"
                  f"- SI-SNR of watermarked vs. original: **{si:.2f} dB**\n"
                  f"- decode on clean watermarked audio: "
                  f"`{_decoded_str(clean_decoded)}` "
                  f"(bit recovery {clean_rec:.3f})")
            per_method_updates += [gr.update(visible=True), gr.update(value=md),
                                   gr.update(value=wav_path),
                                   gr.update(value=png_path)]
            state["wm"][m.key] = {"name": m.name, "bits": bits,
                                  "y": y_wm, "sr": sr_wm, "spec": png_path}
            ok.append(m.name)
        except Exception as e:
            traceback.print_exc()
            per_method_updates += [
                gr.update(visible=True),
                gr.update(value=f"{header}\n\n⚠️ Embedding failed: {e}"),
                gr.update(value=None), gr.update(value=None)]
            failed.append(m.name)
        done += 1

    status = f"✅ Embedded with {len(ok)} method(s): {', '.join(ok) or '—'}."
    if failed:
        status += f"  ⚠️ Failed/unavailable: {', '.join(failed)}."
    if not ok:
        status = "⚠️ No method embedded successfully. " + status
    dur = len(y) / sr
    status += f"  (input: {dur:.1f}s @ {sr} Hz, clipped to ≤{eu.MAX_AUDIO_DURATION:.0f}s)"
    return [status, state if ok else None] + per_method_updates


# ─── Step 2: distort + evaluate ───────────────────────────────────────────────

def do_evaluate(state, selected_dist_labels, *setting_lists,
                progress=gr.Progress()):
    empty_df = gr.update(value=None)
    if not state or not state.get("wm"):
        return ("⚠️ Embed watermarks first (step 1).", empty_df,
                gr.update(choices=[], value=None), state)
    if not selected_dist_labels:
        return ("⚠️ Select at least one distortion type.", empty_df,
                gr.update(choices=[], value=None), state)

    dist_keys = DIST_KEYS
    jobs = []
    for key in state["wm"]:
        for label in selected_dist_labels:
            dkey = CHOICE_TO_KEY[label]
            all_settings = eu.DIGITAL_DISTORTIONS[dkey]
            chosen = setting_lists[dist_keys.index(dkey)]
            settings = [s for s in all_settings if str(s) in set(chosen)]
            for s in settings:
                jobs.append((key, dkey, s))
    if not jobs:
        return ("⚠️ The selected distortion types have no settings selected.",
                empty_df, gr.update(choices=[], value=None), state)
    if len(jobs) > MAX_EVAL_JOBS:
        return (f"⚠️ {len(jobs)} evaluations requested — that exceeds the cap "
                f"of {MAX_EVAL_JOBS}. Deselect some methods, distortions, or "
                f"settings (or raise SOK_DEMO_MAX_JOBS).",
                empty_df, gr.update(choices=[], value=None), state)

    session_dir = Path(state["session_dir"])
    orig_y, orig_sr = state["orig_y"], state["orig_sr"]
    visqol_mode = state.get("visqol_mode", "speech")
    ref_cache = {}
    rows, play_map = [], {}

    for i, (mkey, dkey, setting) in enumerate(jobs):
        info = state["wm"][mkey]
        method = next(m for m in REGISTRY if m.key == mkey)
        y_wm, sr_wm, bits = info["y"], info["sr"], info["bits"]
        label = eu.DISTORTION_LABELS[dkey]
        progress(i / len(jobs),
                 desc=f"[{i + 1}/{len(jobs)}] {info['name']} · {label} = {setting}")
        try:
            y_d = eu.apply_distortion(y_wm, sr_wm, dkey, setting,
                                      stem=f"{state['stem']}:{mkey}")
            y_d = np.asarray(y_d, dtype=np.float32)

            try:
                decoded = method.decode(y_d, sr_wm, len(bits))
            except Exception:
                decoded = None
            rec = eu.bit_recovery(bits, decoded)

            if sr_wm not in ref_cache:
                ref_cache[sr_wm] = librosa.resample(
                    orig_y, orig_sr=orig_sr, target_sr=sr_wm) \
                    if orig_sr != sr_wm else orig_y
            ref_a, deg_a = eu.align_lengths(ref_cache[sr_wm], y_d)
            si = eu.compute_si_snr(ref_a, deg_a)
            visqol = eu.compute_visqol(ref_a, deg_a, sr_wm, mode=visqol_mode)

            slug = eu.bu.setting_label(setting)
            wav_path = _write_wav(
                session_dir / f"{mkey}__{dkey}__{slug}.wav", y_d, sr_wm)
            play_key = f"{info['name']} | {label} = {setting}"
            play_map[play_key] = {"wav": wav_path, "mkey": mkey}

            rows.append([info["name"], label, str(setting), _fmt(rec),
                         _fmt(si, 2), _fmt(visqol), _decoded_str(decoded)])
        except Exception as e:
            traceback.print_exc()
            rows.append([info["name"], label, str(setting), "n/a", "n/a",
                         "n/a", f"error: {e}"])

    state["play_map"] = play_map
    choices = list(play_map)
    status = (f"✅ Evaluated {len(jobs)} distorted watermarked clips "
              f"({len(state['wm'])} method(s) × selected distortion settings). "
              f"Metrics are computed against the clean original; decode "
              f"refusals count as bit recovery 0.0 and show `None`.")
    return (status, gr.update(value=rows),
            gr.update(choices=choices, value=(choices[0] if choices else None)),
            state)


def plan_summary(selected_method_names, selected_dist_labels, *setting_lists):
    """Live preview of how much work the current selection implies."""
    n_methods = len([n for n in (selected_method_names or [])
                     if n in AVAILABLE_NAMES])
    if not selected_dist_labels:
        return ("_No distortion type selected — pick at least one above "
                f"(10 types, {TOTAL_SETTINGS} attack settings in total)._")

    per_type, n_settings = [], 0
    for label in selected_dist_labels:
        dkey = CHOICE_TO_KEY[label]
        chosen = setting_lists[DIST_KEYS.index(dkey)]
        total = len(eu.DIGITAL_DISTORTIONS[dkey])
        n_settings += len(chosen)
        if len(chosen) != total:
            per_type.append(f"{eu.DISTORTION_LABELS[dkey]}: {len(chosen)}/{total}")

    if n_settings == 0:
        return ("⚠️ _The selected distortion types have **no settings** ticked — "
                "open **Attack settings per distortion type** below._")

    jobs = n_methods * n_settings
    msg = (f"**Planned:** {len(selected_dist_labels)} distortion type(s), "
           f"**{n_settings}** attack setting(s) × **{n_methods}** method(s) "
           f"= **{jobs}** evaluation(s).")
    if per_type:
        msg += "  \nNarrowed: " + "; ".join(per_type) + "."
    if jobs > MAX_EVAL_JOBS:
        msg += (f"  \n⚠️ Above the cap of {MAX_EVAL_JOBS} — deselect some "
                f"methods, types, or settings (or raise `SOK_DEMO_MAX_JOBS`).")
    return msg


def do_play(state, play_key):
    """Return the distorted clip plus the three-way spectrogram comparison:
    original upload → watermarked → distorted watermarked."""
    empty = (None, None, None, None)
    if not state or not play_key:
        return empty
    entry = state.get("play_map", {}).get(play_key)
    if not entry:
        return empty
    wav_path = entry["wav"]
    if not Path(wav_path).exists():
        return empty

    # Distorted spectrogram — rendered on demand, then cached next to the wav.
    dist_png = wav_path + ".png"
    if not Path(dist_png).exists():
        try:
            y, sr = sf.read(wav_path)
            if y.ndim > 1:
                y = y.mean(axis=1)
            eu.spectrogram_png(y.astype(np.float32), sr,
                               f"Distorted — {play_key}", dist_png)
        except Exception:
            traceback.print_exc()
            dist_png = None

    info = state.get("wm", {}).get(entry.get("mkey"), {})
    wm_png = info.get("spec")
    if wm_png and not Path(wm_png).exists():
        wm_png = None
    orig_png = state.get("orig_spec")
    if orig_png and not Path(orig_png).exists():
        orig_png = None

    return wav_path, orig_png, wm_png, dist_png


# ─── UI ───────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    unavailable = [m for m in REGISTRY if m.unavailable_reason]
    with gr.Blocks(title="SoK: Audio Watermarking Robustness Demo") as demo:
        gr.Markdown(
            "# Audio Watermarking Robustness — Interactive Demo\n"
            "Companion demo for *“Is Audio Watermarking Robust to Removal "
            "Attacks? A Comprehensive Measurement Study.”*  Upload audio, "
            "embed watermarks with one or more methods, then apply "
            "digital-level distortions and inspect **bit recovery rate**, "
            "**SI-SNR**, **ViSQOL**, and the **decoded payload bits**.\n\n"
            "*All 10 benchmark methods are included. Timbre, AWARE, and "
            "RobustDNN run in isolated worker environments (their "
            "dependencies conflict with the main stack). Note: AWARE embeds "
            "by per-clip optimization and takes minutes on CPU.*"
            + ("\n\n⚠️ Currently unavailable in this container: "
               + ", ".join(f"{m.name} ({m.unavailable_reason})"
                           for m in unavailable) if unavailable else ""))

        session = gr.State()

        gr.Markdown("## 1) Input & watermark embedding")
        with gr.Row():
            audio_in = gr.Audio(label="Input audio (single file, clipped to "
                                      f"{eu.MAX_AUDIO_DURATION:.0f}s)",
                                type="filepath", sources=["upload", "microphone"])
            with gr.Column():
                visqol_mode = gr.Radio(
                    ["speech", "audio"], value="speech",
                    label="ViSQOL mode",
                    info="speech → 16 kHz MOS-LQO; audio (music) → 48 kHz")
                try:
                    gr.Examples([[_make_example_wav()]], inputs=[audio_in],
                                label="Example input (synthetic melody)")
                except Exception:
                    pass

        method_sel = gr.CheckboxGroup(
            choices=METHOD_NAMES, value=AVAILABLE_NAMES,
            label="Watermark methods (select one, several, or all)")
        with gr.Row():
            all_methods_btn = gr.Button("Select all methods", size="sm")
            no_methods_btn = gr.Button("Clear methods", size="sm")

        with gr.Accordion("Watermark payload bits per method "
                          "(defaults match the paper benchmark)", open=False):
            bit_boxes = []
            for m in REGISTRY:
                bit_boxes.append(gr.Textbox(
                    label=f"{m.name} — {m.bits_help}", value=m.default_bits,
                    info=("fixed length" if m.fixed_bits
                          else "variable length within the allowed range")))

        embed_btn = gr.Button("① Embed watermarks", variant="primary")
        embed_status = gr.Markdown()

        method_groups, method_mds, method_audios, method_imgs = [], [], [], []
        for m in REGISTRY:
            with gr.Group(visible=False) as grp:
                md = gr.Markdown()
                with gr.Row():
                    au = gr.Audio(label=f"{m.name} — watermarked audio",
                                  type="filepath", interactive=False)
                    im = gr.Image(label=f"{m.name} — spectrogram",
                                  type="filepath", interactive=False)
            method_groups.append(grp)
            method_mds.append(md)
            method_audios.append(au)
            method_imgs.append(im)

        gr.Markdown("## 2) Digital distortions & robustness evaluation")
        dist_sel = gr.CheckboxGroup(
            choices=DIST_LABELS, value=[],
            label=f"Distortion types — {len(DIST_LABELS)} types, "
                  f"{TOTAL_SETTINGS} attack settings in total (each selected "
                  f"type applies all of its settings unless you narrow them below)")
        with gr.Row():
            all_dist_btn = gr.Button("Select all distortions", size="sm")
            no_dist_btn = gr.Button("Clear distortions", size="sm")

        setting_groups = []
        with gr.Accordion(f"Attack settings per distortion type — tick a subset "
                          f"to test fewer than all {TOTAL_SETTINGS}", open=False):
            gr.Markdown(
                "All settings are selected by default. Untick individual "
                "values to evaluate only a subset of a distortion's settings.")
            with gr.Row():
                all_set_btn = gr.Button("Select all settings", size="sm")
                no_set_btn = gr.Button("Clear all settings", size="sm")
            for key, settings in eu.DIGITAL_DISTORTIONS.items():
                setting_groups.append(gr.CheckboxGroup(
                    choices=[str(s) for s in settings],
                    value=[str(s) for s in settings],
                    label=f"{eu.DISTORTION_LABELS[key]} — {len(settings)} settings"))

        plan_md = gr.Markdown()
        eval_btn = gr.Button("② Apply distortions & evaluate", variant="primary")
        eval_status = gr.Markdown()
        results_df = gr.Dataframe(
            headers=["Method", "Distortion", "Setting", "Bit recovery",
                     "SI-SNR (dB)", "ViSQOL", "Decoded bits"],
            datatype=["str"] * 7, interactive=False, wrap=True,
            label="Per-clip results (distorted watermarked audio vs. clean original)")

        gr.Markdown(
            "### Listen & compare spectrograms\n"
            "Pick an evaluated clip to hear it and compare the three stages "
            "side by side: the uploaded original, the watermarked audio, and "
            "the distorted watermarked audio.")
        play_dd = gr.Dropdown(choices=[], label="Method | distortion = setting")
        play_audio = gr.Audio(label="Distorted watermarked audio",
                              type="filepath", interactive=False)
        with gr.Row():
            spec_orig = gr.Image(label="1. Original (uploaded)",
                                 type="filepath", interactive=False)
            spec_wm = gr.Image(label="2. Watermarked",
                               type="filepath", interactive=False)
            spec_dist = gr.Image(label="3. Distorted watermarked",
                                 type="filepath", interactive=False)

        # ── wiring ──
        all_methods_btn.click(lambda: gr.update(value=METHOD_NAMES),
                              outputs=method_sel)
        no_methods_btn.click(lambda: gr.update(value=[]), outputs=method_sel)
        all_dist_btn.click(lambda: gr.update(value=DIST_LABELS),
                           outputs=dist_sel)
        no_dist_btn.click(lambda: gr.update(value=[]), outputs=dist_sel)
        all_set_btn.click(
            lambda: [gr.update(value=[str(s) for s in v])
                     for v in eu.DIGITAL_DISTORTIONS.values()],
            outputs=setting_groups)
        no_set_btn.click(
            lambda: [gr.update(value=[]) for _ in eu.DIGITAL_DISTORTIONS],
            outputs=setting_groups)

        # Live "how much work is this?" preview.  .change also fires when the
        # select-all / clear buttons update these components programmatically.
        plan_inputs = [method_sel, dist_sel] + setting_groups
        for comp in plan_inputs:
            comp.change(plan_summary, inputs=plan_inputs, outputs=plan_md)
        demo.load(plan_summary, inputs=plan_inputs, outputs=plan_md)

        per_method_outputs = []
        for g, md, au, im in zip(method_groups, method_mds,
                                 method_audios, method_imgs):
            per_method_outputs += [g, md, au, im]
        embed_btn.click(
            do_embed,
            inputs=[audio_in, method_sel, visqol_mode] + bit_boxes,
            outputs=[embed_status, session] + per_method_outputs)

        eval_btn.click(
            do_evaluate,
            inputs=[session, dist_sel] + setting_groups,
            outputs=[eval_status, results_df, play_dd, session],
        ).then(  # populate the comparison view for the first clip right away
            do_play, inputs=[session, play_dd],
            outputs=[play_audio, spec_orig, spec_wm, spec_dist])

        play_dd.change(do_play, inputs=[session, play_dd],
                       outputs=[play_audio, spec_orig, spec_wm, spec_dist])
    return demo


def main():
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        allowed_paths=[str(TMP_ROOT)],
        show_error=True)


if __name__ == "__main__":
    main()
