# Cinderworks Studio

A local-first image generation studio with a Gradio web UI. Phase 1 ships the core creative loop: download Krea 2 Turbo, generate an image, persist the job, and recall history.

**Product thesis:** "Works when you log in. Still works tomorrow."

## Features

- One-click install via bootstrap script (Windows & Linux)
- Krea 2 Turbo image generation (8-step Turbo mode, guidance disabled)
- In-app model download with streaming progress and resume-on-interrupt
- VRAM tenant discipline — automatic GPU memory management on 24 GB cards
- Full generation history with parameter recall, reproduction, and deletion
- Glassmorphism UI with lemon/amber accents
- 226 tests (property-based + unit) covering all modules

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | 3.11, 3.12, or 3.13 all work |
| Git | any | For cloning |
| uv | any | Python package installer — [install guide](https://docs.astral.sh/uv/getting-started/installation/) |
| NVIDIA GPU | 24 GB VRAM recommended | RTX 4090, 3090, etc. |
| CUDA | 12.x | Matching your torch build |
| hf (optional) | any | HuggingFace CLI for faster model downloads |

## Quick Start

### Windows

```cmd
git clone https://github.com/sjfischr/cinderworks.git
cd cinderworks\studio
install\bootstrap.bat
```

The bootstrap script creates a local `.venv`, installs PyTorch with CUDA support, installs all other dependencies, and launches the Gradio server. Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

After initial setup, use the quick launcher:

```cmd
launch.bat
```

### Linux

```bash
git clone https://github.com/sjfischr/cinderworks.git
cd cinderworks/studio
chmod +x install/bootstrap.sh launch.sh
./install/bootstrap.sh
```

After initial setup:

```bash
./launch.sh
```

### Manual Setup

```cmd
cd cinderworks\studio
uv venv .venv --python python
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

uv pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install -r requirements.txt

copy .env.example .env        # adjust paths if needed
python app.py
```

## Model Weights

Cinderworks uses the `Krea2Pipeline` from diffusers for inference. The diffusers-format weights need to be downloaded once (~36 GB). The fastest method is via the HuggingFace CLI:

```cmd
hf download krea/Krea-2-Turbo --local-dir models_store/krea2-turbo-diffusers
```

If you skip this step, the pipeline will auto-download from HuggingFace on first generate (slower, unauthenticated rate limits apply).

The in-app "Download Krea 2 Turbo" button downloads the Comfy-Org safetensors (used for readiness status checking). These are separate from the diffusers-format weights used for inference. A future update will consolidate to a single set.

## First Run

1. **Launch** — the UI opens immediately (no model loading at startup).
2. **Download models** — either via `hf download` (recommended, see above) or the pipeline auto-downloads on first generate.
3. **Generate tab** → type a prompt, click Generate. First generation loads the pipeline (~30s), subsequent ones are faster.
4. **History tab** → past generations are saved with full parameters for reproduction. You can load params from any past job or delete jobs.

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
├── launch.bat          # Quick launch (Windows)
├── launch.sh           # Quick launch (Linux)
├── requirements.txt    # Dependencies (torch installed separately with CUDA)
├── uv.toml             # uv package manager config (CUDA index)
├── .env.example        # Template config
├── core/
│   ├── model_loader.py # Lazy loading, caches by (model_id, precision)
│   ├── system_check.py # CUDA/model readiness detection
│   └── vram_manager.py # Single-tenant GPU memory coordinator
├── db/
│   └── db.py           # SQLite CRUD (job + artifact tables, delete support)
├── models/
│   ├── registry.py     # Model-agnostic routing layer
│   ├── downloader.py   # Streaming, resumable HuggingFace downloads
│   └── backends/
│       └── krea2.py    # Krea 2 Turbo via diffusers Krea2Pipeline
├── ui/
│   ├── theme.py        # Glassmorphism CSS + Gradio theme
│   ├── controls.py     # Parameter components + validation
│   └── handlers.py     # Error boundary, event handlers
├── install/
│   ├── bootstrap.bat   # Windows one-click install + launch
│   └── bootstrap.sh    # Linux one-click install + launch
├── tests/              # pytest + hypothesis (226 tests)
├── models_store/       # Downloaded model weights (git-ignored)
│   └── krea2-turbo-diffusers/  # Diffusers-format weights (via hf download)
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

All tests run without a GPU — model operations are mocked when CUDA is unavailable.

## Architecture Decisions

- **Shell stays thin** — `app.py` wires components. No inference, download, or SQL.
- **Registry indirection** — UI never imports backends directly. Adding a model = one registry entry + one backend module.
- **Single GPU chokepoint** — only `vram_manager.py` coordinates GPU tenancy. The pipeline internally handles component offloading.
- **Error boundary at UI** — handlers catch everything, log tracebacks to file, show plain-language messages to the user.
- **Lazy loading** — models load on first generate, not on boot. UI is navigable in under 5 seconds.
- **Dual-path backend** — real inference (Krea2Pipeline + CUDA) or stub (no GPU, for tests). Auto-detected.

## Known Limitations (Phase 1)

- Diffusers-format weights and Comfy-Org safetensors are separate downloads (will be consolidated)
- `Krea2Pipeline` is installed from diffusers `main` branch (not yet in a stable PyPI release)
- bf16 precision requires ~24 GB VRAM (tight fit on RTX 4090 with other processes running)

## License

MIT — see [LICENSE](LICENSE) for details.
