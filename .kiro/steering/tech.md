---
inclusion: auto
---

# Technology Stack — Cinderworks

> Authoritative technology choices. When implementing, follow these constraints.

## Language & Runtime

- **Python 3.11+** (3.11, 3.12, or 3.13 all work; torch 2.7+ has wheels for all).
- Manage the environment with **`uv`** (fast, reproducible) creating a
  project-local `venv`.
- Never let the app auto-recreate or auto-upgrade its own environment at launch.
- Target OS: **Windows** primary, Linux secondary. Keep paths `pathlib`-clean.

## UI

- **Gradio** (Blocks API) — firm choice, not a default. The product thesis is
  "Gradio-style WebUI, not node graph."
- Custom CSS for glassmorphism theme. No heavyweight front-end framework.

## ML / Inference Stack

- **PyTorch** with CUDA (target: RTX 4090, 24 GB). Default dtype **bfloat16**.
- Precision choices: `bf16` (fits 24 GB), `fp8_scaled` (~13 GB).
- **diffusers** + **transformers** for model components.
- **huggingface_hub** for model downloads (streaming, resumable).
- Attention: prefer PyTorch native `scaled_dot_product_attention`. Optional
  SageAttention/FlashAttention/xformers behind feature flags. Never hard-fail
  if an optional attention backend is missing.

## Krea 2 Canonical Files

- Diffusion (Turbo): `krea2_turbo_fp8_scaled.safetensors` (~13 GB) or
  `krea2_turbo_bf16.safetensors` (~25 GB).
- Text encoder: `qwen3vl_4b_fp8_scaled.safetensors`.
- VAE: `qwen_image_vae.safetensors`.
- Turbo sampler defaults: **8 steps, guidance_scale 0.0 (disabled per Krea convention), fixed mu/shift 1.15**.

## Persistence

- **SQLite** via stdlib `sqlite3`. One DB file (`studio.db`). No ORM. Plain
  parameterized SQL in `db/db.py`.
- Artifacts (images) in `outputs/job_<id>/`, paths recorded in DB.

## Config

- **python-dotenv** `.env` driven. Keys: `MODEL_DIR`, `OUTPUT_DIR`, `APP_NAME`,
  `DB_PATH`. Provide `.env.example`.
- A central `Config` object resolves all paths off a single base dir.

## Dependency Discipline

- **Pin where stable, floor where evolving.** Core UI and testing deps use exact
  `==` versions. ML deps (diffusers, transformers, safetensors, accelerate) use
  `>=` minimums because the Krea 2 pipeline is on diffusers `main` branch and
  its transitive deps evolve rapidly.
- **Torch is installed separately** from the CUDA index by the bootstrap script
  (not from PyPI which is CPU-only).
- **No auto-self-update.** The app never `git pull`s or `pip install`s itself on
  launch.
- **Decouple model backends from the shell.** A backend importing a broken dep
  must degrade to disabled-with-reason, never crash the app.

## Testing

- **pytest.** Every pattern module ships with tests. Test behavior (EARS
  acceptance criteria) not implementation detail.

## Explicit Non-Choices

- No node-graph engine. No A1111/Forge codebase fork. No ORM in Phase 1.
- No web framework beyond Gradio. No auto-updater. No bundled weights.
