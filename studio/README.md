# Cinderworks Studio

A local-first image generation studio with a Gradio web UI. Phase 1 ships the core creative loop: download Krea 2 Turbo, generate an image, persist the job, and recall history.

**Product thesis:** "Works when you log in. Still works tomorrow."

## Features

- One-click install via bootstrap script (Windows & Linux)
- Krea 2 Turbo image generation (8-step Turbo mode, bf16 or fp8_scaled precision)
- In-app model download with streaming progress and resume-on-interrupt
- VRAM tenant discipline — automatic GPU memory management on 24 GB cards
- Full generation history with parameter recall and reproduction
- Glassmorphism UI with lemon/amber accents
- 226 tests (property-based + unit) covering all modules

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11.x | Not 3.12+ (torch pinned to 3.11-compatible build) |
| Git | any | For cloning |
| uv | any | Python package installer — [install guide](https://docs.astral.sh/uv/getting-started/installation/) |
| NVIDIA GPU | 24 GB VRAM recommended | RTX 4090, 3090, etc. fp8_scaled works with ~13 GB |
| CUDA | 12.x | Matching your torch build |

## Quick Start

### Windows

```cmd
git clone <your-repo-url> cinderworks
cd cinderworks\studio
install\bootstrap.bat
```

The script creates a local `.venv`, installs exact-pinned dependencies, and launches the Gradio server. Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

### Linux

```bash
git clone <your-repo-url> cinderworks
cd cinderworks/studio
chmod +x install/bootstrap.sh
./install/bootstrap.sh
```

### Manual Setup (if bootstrap doesn't suit you)

```cmd
cd cinderworks\studio
uv venv .venv --python python3.11
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

copy .env.example .env        # adjust paths if needed
python app.py
```

## First Run

1. **Launch** — the UI opens immediately (no model loading at startup).
2. **Models tab** → click "Download Krea 2 Turbo". Progress streams live.
3. Once downloaded, the readiness banner disappears.
4. **Generate tab** → type a prompt, click Generate. Images appear in the gallery.
5. **History tab** → past generations are saved with full parameters for reproduction.

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
APP_NAME=Cinderworks
MODEL_DIR=models_store     # where model weights live
OUTPUT_DIR=outputs         # where generated images are saved
DB_PATH=studio.db          # SQLite database location
```

All paths are relative to `studio/` unless absolute.

## Project Structure

```
studio/
├── app.py              # Gradio Blocks shell (thin — wiring only)
├── config.py           # Config from .env
├── requirements.txt    # Exact-pinned dependencies
├── .env.example        # Template config
├── core/
│   ├── model_loader.py # Lazy loading, caches by (model_id, precision)
│   ├── system_check.py # CUDA/model readiness detection
│   └── vram_manager.py # Single-tenant GPU memory coordinator
├── db/
│   └── db.py           # SQLite CRUD (job + artifact tables)
├── models/
│   ├── registry.py     # Model-agnostic routing layer
│   ├── downloader.py   # Streaming, resumable HuggingFace downloads
│   └── backends/
│       └── krea2.py    # Krea 2 Turbo generation pipeline
├── ui/
│   ├── theme.py        # Glassmorphism CSS + Gradio theme
│   ├── controls.py     # Parameter components + validation
│   └── handlers.py     # Error boundary, event handlers
├── install/
│   ├── bootstrap.bat   # Windows one-click
│   └── bootstrap.sh    # Linux one-click
├── tests/              # pytest + hypothesis (226 tests)
├── models_store/       # Downloaded model weights (git-ignored)
└── outputs/            # Generated images (git-ignored)
```

## Running Tests

```cmd
cd cinderworks\studio
..\.venv\Scripts\python -m pytest tests/ -v
```

Or from the repo root:

```cmd
.venv\Scripts\python -m pytest studio/tests/ -v
```

All tests run without a GPU — model operations are mocked.

## Architecture Decisions

- **Shell stays thin** — `app.py` wires components. No inference, download, or SQL.
- **Registry indirection** — UI never imports backends directly. Adding a model = one registry entry + one backend module.
- **Single GPU chokepoint** — only `vram_manager.py` moves tensors to/from GPU.
- **Error boundary at UI** — handlers catch everything, log tracebacks to file, show plain-language messages to the user.
- **Lazy loading** — models load on first generate, not on boot. UI is navigable in under 5 seconds.

## License

Private / proprietary. All rights reserved.
