#!/usr/bin/env bash
# ==============================================================================
#  Cinderworks Studio — One-Click Bootstrap (Linux/macOS)
#
#  Checks prerequisites (Python 3.11, Git, uv), creates a project-local venv,
#  installs exact-pinned dependencies, and launches the Gradio server.
#
#  If a prerequisite is missing, reports which one and exits non-zero
#  WITHOUT creating or modifying the virtual environment.
#
#  This script does NOT perform git pull or self-directed pip install.
# ==============================================================================

set -euo pipefail

# --- Resolve paths relative to the studio root (parent of install/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$STUDIO_ROOT/.venv"
REQUIREMENTS="$STUDIO_ROOT/requirements.txt"

# ==============================================================================
#  Prerequisite checks — all must pass before any environment work
# ==============================================================================

# --- Check Python 3.11 ---
if ! command -v python3 &>/dev/null; then
    echo "[Cinderworks] ERROR: Missing prerequisite: Python 3.11"
    echo "[Cinderworks] Please install Python 3.11 and try again."
    exit 1
fi

PY_VERSION="$(python3 --version 2>&1)"
if [[ "$PY_VERSION" != *"3.11"* ]]; then
    echo "[Cinderworks] ERROR: Missing prerequisite: Python 3.11 (found: $PY_VERSION)"
    echo "[Cinderworks] Please install Python 3.11 and try again."
    exit 1
fi

# --- Check Git ---
if ! command -v git &>/dev/null; then
    echo "[Cinderworks] ERROR: Missing prerequisite: Git"
    echo "[Cinderworks] Please install Git and try again."
    exit 1
fi

# --- Check uv ---
if ! command -v uv &>/dev/null; then
    echo "[Cinderworks] ERROR: Missing prerequisite: uv"
    echo "[Cinderworks] Please install uv and try again."
    exit 1
fi

# ==============================================================================
#  All prerequisites met — create venv and install
# ==============================================================================

echo "[Cinderworks] All prerequisites found."
echo "[Cinderworks] Creating project-local virtual environment..."

uv venv "$VENV_DIR" --python python3
if [ $? -ne 0 ]; then
    echo "[Cinderworks] ERROR: Failed to create virtual environment."
    exit 1
fi

echo "[Cinderworks] Installing pinned dependencies from requirements.txt..."

uv pip install --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS"
if [ $? -ne 0 ]; then
    echo "[Cinderworks] ERROR: Failed to install dependencies."
    exit 1
fi

# ==============================================================================
#  Launch the Gradio server
# ==============================================================================

echo "[Cinderworks] Launching Cinderworks Studio..."
exec "$VENV_DIR/bin/python" "$STUDIO_ROOT/app.py"
