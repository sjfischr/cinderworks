---
inclusion: auto
---

# Project Structure — Cinderworks

> Authoritative directory layout and module boundary rules. This layout is
> derived from the proven BeatBunny/Higgs Studio template. These boundaries were
> arrived at through test-driven iteration and should be reproduced exactly.

## Directory Layout

```
studio/
├── app.py                      # Gradio Blocks shell: tabs, CSS, event wiring.
│                               #   Thin — delegates all logic to modules below.
├── config.py                   # Config object; resolves paths from .env
├── .env.example                # MODEL_DIR, OUTPUT_DIR, APP_NAME, DB_PATH
├── requirements.txt            # exact-pinned deps
├── ui/
│   ├── theme.py                # glassmorphism CUSTOM_CSS
│   ├── handlers.py             # Gradio event handlers; wrap workers, surface
│   │                           #   plain-language errors, never leak tracebacks
│   └── controls.py             # reusable control groups (sampler params, batch,
│                               #   precision picker)
├── core/
│   ├── system_check.py         # check_cuda_status, check_model_status,
│   │                           #   get_system_status_text, is_ready_to_generate,
│   │                           #   get_readiness_banner
│   ├── model_loader.py         # lazy load; loads on first generate, not boot
│   └── vram_manager.py         # tenant registration; load→use→unload discipline
├── models/
│   ├── registry.py             # model-agnostic registry; Phase 1 = 1 entry
│   ├── downloader.py           # streaming, resumable HF downloader (generator)
│   └── backends/
│       └── krea2.py            # Krea 2 backend: encode, sample, decode
├── db/
│   └── db.py                   # init_db, create_job, get_recent_jobs, get_job,
│                               #   get_job_artifacts (plain sqlite3, no ORM)
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

## Module Boundary Rules (Enforced)

1. **`app.py` stays thin.** It wires Gradio components to handlers. No inference,
   no download, no SQL in `app.py`.
2. **The shell never imports a model backend directly.** It goes through
   `models/registry.py`. Backends are imported lazily and guarded.
3. **Only `core/vram_manager.py` moves models on/off the GPU.** Nothing else
   calls `.to('cuda')`/`.to('cpu')` directly.
4. **Only `db/db.py` touches SQLite.** Handlers call db functions; they don't
   write SQL inline.
5. **Errors surface as text, not exceptions, at the UI boundary.** Handlers
   catch, log, and return a `❌ <plain language>` string.

## Naming Conventions

- Files and modules: `snake_case.py`.
- DB tables: `snake_case`, singular-noun rows (`job`, `artifact`).
- Job output dirs: `outputs/job_<zero-padded-id>/`.
- Model registry ids: short lowercase (`krea2-turbo`).
- A single `APP_NAME` constant in `config.py` is the only place the product
  name lives.

## Anti-Patterns to Avoid

- ❌ A model backend that raises on import and takes the whole UI down.
- ❌ Inference code that unloads/loads models itself instead of asking the
  vram_manager.
- ❌ A raw traceback rendered in the Gradio UI.
- ❌ Any `git pull` / `pip install` triggered from application code at runtime.
- ❌ Hidden generation params that aren't exposed as UI controls.
