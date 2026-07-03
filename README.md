# Cinderworks Studio

A local-first image generation studio with a Gradio web UI. Phase 1 ships the core creative loop: download Krea 2 Turbo, generate an image, persist the job, and recall history.

**Product thesis:** "Works when you log in. Still works tomorrow."

## Features

- One-click install via bootstrap script (Windows & Linux)
- Krea 2 Turbo image generation (8-step Turbo mode, bf16 or fp8_scaled precision)
- Real inference via diffusers `Krea2Pipeline` with automatic model CPU offload
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
| NVIDIA GPU | 24 GB VRAM recommended | RTX 4090, 3090, etc. fp8_scaled works with ~13 GB |
| CUDA | 12.x | Matching your torch build |

## Quick Start

### Windows

```cmd
git clone https://github.com/sjfischr/cinderworks.git
cd cinderworks\studio
install\bootstrap.bat
```

The script creates a local `.venv`, installs PyTorch with CUDA + all dependencies, and launches the Gradio server. Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

### Linux

```bash
git clone https://github.com/sjfischr/cinderworks.git
cd cinderworks/studio
chmod +x install/bootstrap.sh
./install/bootstrap.sh
```

### Quick Launch (after initial setup)

```cmd
cd cinderworks\studio
launch.bat          # Windows
./launch.sh         # Linux
```

### Manual Setup

```cmd
cd cinderworks\studio
uv venv .venv --python python
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

copy .env.example .env        # adjust paths if needed
python app.py
```

## First Run

1. **Launch** — the UI opens immediately (no model loading at startup).
2. **Download model weights** — you have two options:
   - **In-app:** Models tab → click "Download Krea 2 Turbo" (downloads Comfy-Org safetensors for readiness tracking)
   - **CLI (faster with HF token):** `hf download krea/Krea-2-Turbo --local-dir models_store/krea2-turbo-diffusers`
3. **Generate** — type a prompt, click Generate. First generation loads the pipeline (~30s), subsequent ones are fast.
4. **History** — past generations are saved with full parameters for reproduction.

## Model Weights

The app uses the diffusers-format weights from `krea/Krea-2-Turbo` on HuggingFace (~36 GB). These can be:
- Auto-downloaded on first generation (goes to HF cache at `~/.cache/huggingface/`)
- Pre-downloaded to `models_store/krea2-turbo-diffusers/` via:
  ```cmd
  hf download krea/Krea-2-Turbo --local-dir models_store/krea2-turbo-diffusers
  ```

The in-app downloader downloads the Comfy-Org format safetensors (used for readiness status display).

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
├── launch.bat/.sh      # Quick-start scripts
├── requirements.txt    # Dependencies
├── .env.example        # Template config
├── core/
│   ├── model_loader.py # Lazy loading, caches by (model_id, precision)
│   ├── system_check.py # CUDA/model readiness detection
│   └── vram_manager.py # Single-tenant GPU memory coordinator
├── db/
│   └── db.py           # SQLite CRUD (job + artifact tables + delete)
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
└── outputs/            # Generated images (git-ignored)
```

## Running Tests

```cmd
cd cinderworks\studio
..\.venv\Scripts\python -m pytest tests/ -v
```

All tests run without a GPU — model operations are mocked when CUDA is unavailable.

## Architecture Decisions

- **Shell stays thin** — `app.py` wires components. No inference, download, or SQL.
- **Registry indirection** — UI never imports backends directly. Adding a model = one registry entry + one backend module.
- **Diffusers pipeline** — real inference uses `Krea2Pipeline` with `model_cpu_offload()` for automatic VRAM management.
- **Error boundary at UI** — handlers catch everything, log tracebacks to file, show plain-language messages to the user.
- **Lazy loading** — models load on first generate, not on boot. UI is navigable in under 5 seconds.
- **Turbo defaults** — 8 steps, guidance_scale=0.0 (disabled), mu/shift=1.15.

## License

MIT — see [LICENSE](LICENSE) for details.
