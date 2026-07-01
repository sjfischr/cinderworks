# Project Structure

This layout is derived directly from the proven BeatBunny/Higgs Studio template.
It is not a suggestion to improve on — these boundaries were arrived at through
test-driven iteration and should be reproduced. See the patterns steering doc
for why each module exists.

## Directory layout

```
studio/
├── app.py                      # Gradio Blocks shell: tabs, CSS, event wiring.
│                               #   Thin — delegates all logic to modules below.
├── config.py                   # Config object; resolves paths from .env
├── .env.example                # MODEL_DIR, OUTPUT_DIR, APP_NAME, DB_PATH
├── requirements.txt            # exact-pinned deps
├── ui/
│   ├── theme.py                # glassmorphism CUSTOM_CSS (see patterns doc)
│   ├── handlers.py             # Gradio event handlers; wrap workers, surface
│   │                           #   plain-language errors, never leak tracebacks
│   └── controls.py             # reusable control groups (sampler params, batch,
│                               #   precision picker). Params surfaced as real
│                               #   UI controls, not hidden flags (Higgs lesson)
├── core/
│   ├── system_check.py         # check_cuda_status, check_model_status,
│   │                           #   get_system_status_text, is_ready_to_generate,
│   │                           #   get_readiness_banner
│   ├── model_loader.py         # lazy load; loads on first generate, not boot
│   └── vram_manager.py         # tenant registration; load→use→unload discipline
├── models/
│   ├── registry.py             # model-agnostic registry; Phase 1 = 1 entry
│   ├── downloader.py           # streaming, resumable HF downloader (generator
│   │                           #   that yields progress); auto-placement
│   └── backends/
│       └── krea2.py            # Krea 2 backend: load DiT/VAE/Qwen3-VL, encode,
│                               #   sample (euler flow + optional CFG), decode
├── db/
│   └── db.py                   # init_db, create_job, get_recent_jobs, get_job,
│                               #   get_job_artifacts  (plain sqlite3, no ORM)
├── install/
│   ├── bootstrap.bat           # Windows one-click: uv venv, pinned install, run
│   └── bootstrap.sh            # Linux/Mac equivalent
├── outputs/                    # generated images: outputs/job_<id>/*.png
├── models_store/               # downloaded weights (== MODEL_DIR default)
└── tests/
    ├── test_downloader.py
    ├── test_system_check.py
    ├── test_db.py
    ├── test_vram_manager.py
    ├── test_registry.py
    └── test_krea2_backend.py
```

## Module boundary rules (enforced, not aspirational)

1. **`app.py` stays thin.** It wires Gradio components to handlers. No inference,
   no download, no SQL in `app.py`. If logic is growing there, it belongs in a
   module.
2. **The shell never imports a model backend directly.** It goes through
   `models/registry.py`. Backends are imported lazily and guarded, so a broken
   backend disables itself (with a reason shown in the UI) instead of crashing
   the app. This is the decoupling requirement from Layer 4 §6.1.
3. **Only `core/vram_manager.py` moves models on/off the GPU.** Nothing else
   calls `.to('cuda')`/`.to('cpu')` directly. Every GPU tenant (image backend,
   and later the prompt LLM and trainer subprocess) registers with it. This is
   how the "aggressive unload" guarantee stays true as tenants are added.
4. **Only `db/db.py` touches SQLite.** Handlers call db functions; they don't
   write SQL inline.
5. **Errors surface as text, not exceptions, at the UI boundary.** Handlers
   catch, log, and return a `❌ <plain language>` string. Tracebacks go to the
   log file, never the user's screen. (Directly answers the ComfyUI red-error
   pain, Layer 1 §3.3.)

## Naming conventions

- Files and modules: `snake_case.py`. Kiro custom steering files:
  `kebab-case.md`.
- DB tables: `snake_case`, singular-noun rows (`job`, `artifact`).
- Job output dirs: `outputs/job_<zero-padded-id>/`.
- Model registry ids: short lowercase (`krea2-turbo`, later `flux-dev`).
- A single `APP_NAME` constant in `config.py` is the only place the product
  name lives.

## What good looks like (anti-patterns to avoid)

- ❌ A model backend that raises on import and takes the whole UI down.
- ❌ Inference code that unloads/loads models itself instead of asking the
  vram_manager.
- ❌ A raw traceback rendered in the Gradio UI.
- ❌ Any `git pull` / `pip install` triggered from application code at runtime.
- ❌ Hidden generation params that aren't exposed as UI controls (unless the
  model bakes them in, like Krea's prompt template — document those).
