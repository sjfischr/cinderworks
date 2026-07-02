#!/usr/bin/env bash
# Cinderworks Studio — Quick Launch (Linux/macOS)
# Activates the venv and starts the Gradio server.

STUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$STUDIO_ROOT/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[Cinderworks] No .venv found. Run install/bootstrap.sh first."
    exit 1
fi

echo "[Cinderworks] Launching..."
exec "$VENV_DIR/bin/python" "$STUDIO_ROOT/app.py"
