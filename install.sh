#!/usr/bin/env bash
#
# One-command setup for the audio-watermarking robustness demo.
#
#   ./install.sh                 install Docker if needed, build, and launch
#   ./install.sh --port 8080     serve the demo on a different host port
#   ./install.sh --rebuild       force a clean image rebuild (--no-cache)
#   ./install.sh --build-only    build the image but do not start the demo
#
# After the script finishes, open http://localhost:7860 in your browser.

set -euo pipefail

IMAGE_NAME="sok-audio-watermark-demo"
CONTAINER_NAME="sok-audio-watermark-demo"
PORT=7860
REBUILD=0
BUILD_ONLY=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)       PORT="${2:?--port needs a value}"; shift 2 ;;
        --rebuild)    REBUILD=1; shift ;;
        --build-only) BUILD_ONLY=1; shift ;;
        -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            die "Unknown option: $1 (see --help)" ;;
    esac
done

# ── sudo helper ──────────────────────────────────────────────────────────────
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        warn "Not running as root and sudo is unavailable — Docker installation/startup may fail."
    fi
fi

# ── 1. Install Docker if missing ─────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    OS="$(uname -s)"
    if [[ "$OS" == "Darwin" ]]; then
        die "Docker is not installed. On macOS, install Docker Desktop from https://docs.docker.com/desktop/setup/install/mac-install/ and re-run this script."
    elif [[ "$OS" == "Linux" ]]; then
        log "Docker not found — installing via https://get.docker.com ..."
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO /tmp/get-docker.sh https://get.docker.com
        else
            die "Neither curl nor wget is available; install one of them (or Docker itself) and re-run."
        fi
        $SUDO sh /tmp/get-docker.sh
        rm -f /tmp/get-docker.sh
        log "Docker installed."
    else
        die "Unsupported OS '$OS'. Install Docker manually and re-run this script."
    fi
fi

# ── 2. Make sure the Docker daemon is running and reachable ──────────────────
DOCKER="docker"
if ! $DOCKER info >/dev/null 2>&1; then
    if [[ -n "$SUDO" ]] && $SUDO docker info >/dev/null 2>&1; then
        DOCKER="$SUDO docker"
        warn "Using 'sudo docker' (add your user to the 'docker' group to avoid this: sudo usermod -aG docker \$USER)"
    else
        log "Docker daemon not reachable — trying to start it ..."
        $SUDO systemctl start docker 2>/dev/null || $SUDO service docker start 2>/dev/null || true
        sleep 3
        if $DOCKER info >/dev/null 2>&1; then
            :
        elif [[ -n "$SUDO" ]] && $SUDO docker info >/dev/null 2>&1; then
            DOCKER="$SUDO docker"
        else
            die "Could not reach the Docker daemon. Start it manually (e.g. 'sudo systemctl start docker' or launch Docker Desktop) and re-run."
        fi
    fi
fi

# ── 3. Build the image (installs all Python/system dependencies inside) ─────
BUILD_ARGS=()
[[ "$REBUILD" -eq 1 ]] && BUILD_ARGS+=(--no-cache)
log "Building Docker image '$IMAGE_NAME' (first build downloads model checkpoints; this can take a while) ..."
$DOCKER build "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" -t "$IMAGE_NAME" "$SCRIPT_DIR"
log "Image built."

if [[ "$BUILD_ONLY" -eq 1 ]]; then
    log "Build-only mode — start the demo later with:"
    log "  $DOCKER run --rm -p ${PORT}:7860 --name $CONTAINER_NAME $IMAGE_NAME"
    exit 0
fi

# ── 4. Start (or restart) the demo container ─────────────────────────────────
if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Removing existing container '$CONTAINER_NAME' ..."
    $DOCKER rm -f "$CONTAINER_NAME" >/dev/null
fi
log "Starting demo container on port $PORT ..."
$DOCKER run -d --name "$CONTAINER_NAME" -p "${PORT}:7860" \
    --restart unless-stopped "$IMAGE_NAME" >/dev/null

# ── 5. Wait until the Gradio server answers ──────────────────────────────────
log "Waiting for the demo to come up (model loading can take ~1 minute) ..."
for i in $(seq 1 60); do
    if command -v curl >/dev/null 2>&1 \
        && curl -fsS "http://localhost:${PORT}/" >/dev/null 2>&1; then
        break
    fi
    if ! $DOCKER ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        $DOCKER logs "$CONTAINER_NAME" 2>&1 | tail -30 || true
        die "The demo container exited unexpectedly — see the log excerpt above ('$DOCKER logs $CONTAINER_NAME' for more)."
    fi
    sleep 3
done

log ""
log "✅ Demo is running:  http://localhost:${PORT}"
log ""
log "Useful commands:"
log "  $DOCKER logs -f $CONTAINER_NAME     # follow server logs"
log "  $DOCKER rm -f $CONTAINER_NAME       # stop and remove the demo"
log "  ./install.sh --rebuild              # rebuild from scratch"
