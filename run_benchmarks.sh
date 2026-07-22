#!/usr/bin/env bash
#
# Run the full robustness benchmark: every watermarking method x every
# distortion setting x every dataset present under SOK_DATASET_DIR.
#
#   ./run_benchmarks.sh                     run all 10 methods
#   ./run_benchmarks.sh --methods audioseal,kosta
#   ./run_benchmarks.sh --skip-preflight
#
# Prerequisites (see requirements/README.md):
#   - per-method virtualenvs under envs/ (override root with SOK_ENVS_DIR)
#   - SOK_DATASET_DIR pointing at the sampled dataset tree
#     (build it with scripts/16_sample_dataset.py)
#   - SOK_NOISE_DIR / SOK_RIR_DIR for the background-noise and reverberation
#     distortions, SOK_AUDIOWMARK_BIN for audiowmark
#   - a CUDA GPU (the AI-based runners require one)
#
# Each runner is resume-safe: re-run this script after an interruption and
# completed datasets/files are skipped.  Results land in
# results/benchmark/{dataset_key}/{algorithm}.json; logs in results/logs/.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="${SOK_ENVS_DIR:-$SCRIPT_DIR/envs}"
LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR"

METHODS="audioseal,wavmark,silentcipher,timbre,dnn,aware,audiowmark,kosta"
SKIP_PREFLIGHT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)        METHODS="${2:?--methods needs a value}"; shift 2 ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

# method -> "env_name:runner_script"
declare -A RUNNERS=(
    [audioseal]="audioseal:scripts/13_large_audioseal.py"
    [wavmark]="wavmark:scripts/13_large_wavmark.py"
    [silentcipher]="silentcipher:scripts/13_large_silentcipher.py"
    [timbre]="timbre:scripts/13_large_timbre.py"
    [dnn]="dnn_wm:scripts/13_large_dnn.py"
    [aware]="aware:scripts/13_large_aware.py"
    [audiowmark]="viz:scripts/13_large_audiowmark.py"
    [kosta]="kosta:scripts/13_large_kosta.py"     # FSVC + Patchwork + Norm-space
)
ORDER=(audioseal wavmark silentcipher timbre dnn aware audiowmark kosta)

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
    PREFLIGHT_PY="$ENVS_DIR/viz/bin/python"
    [[ -x "$PREFLIGHT_PY" ]] || PREFLIGHT_PY="python3"
    echo "== Preflight (config + corpora paths) =="
    "$PREFLIGHT_PY" "$SCRIPT_DIR/scripts/preflight.py" || {
        echo "Preflight reported problems. Fix them or re-run with --skip-preflight." >&2
        exit 1
    }
fi

IFS=',' read -r -a SELECTED <<< "$METHODS"
declare -A WANT=()
for m in "${SELECTED[@]}"; do WANT[$m]=1; done

PASS=() FAIL=() SKIP=()
for m in "${ORDER[@]}"; do
    [[ -n "${WANT[$m]:-}" ]] || continue
    IFS=':' read -r env_name runner <<< "${RUNNERS[$m]}"
    PY="$ENVS_DIR/$env_name/bin/python"
    if [[ ! -x "$PY" ]]; then
        echo "== $m: SKIPPED (missing env: $PY — see requirements/${env_name}.txt) =="
        SKIP+=("$m")
        continue
    fi
    log="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${m}.log"
    echo "== $m: $PY $runner (log: $log) =="
    if "$PY" "$SCRIPT_DIR/$runner" 2>&1 | tee "$log"; then
        PASS+=("$m")
    else
        echo "== $m FAILED — see $log ==" >&2
        FAIL+=("$m")
    fi
done

echo
echo "==================== Benchmark summary ===================="
echo "  completed: ${PASS[*]:-none}"
echo "  failed:    ${FAIL[*]:-none}"
echo "  skipped:   ${SKIP[*]:-none}"
echo "  results:   results/benchmark/{dataset_key}/{algorithm}.json"
echo
echo "Build per-dataset Excel workbooks with, e.g.:"
echo "  python3 scripts/18_dataset_full_excel.py --dataset speech_ljspeech \\"
echo "      --out results/speech_ljspeech_full.xlsx --suffixes __plain__"

[[ ${#FAIL[@]} -eq 0 ]] || exit 1
