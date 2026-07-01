# Technology Stack

## Language & runtime

- **Python 3.11** (pin to 3.11.x; matches the Forge/Krea ecosystem and avoids
  3.12+ wheel gaps for ML deps). Manage the environment with **`uv`** (fast,
  reproducible) creating a project-local `venv`. Never let the app auto-recreate
  or auto-upgrade its own environment at launch.
- Target OS: **Windows** primary (owner's machine, Steve-i9), Linux secondary.
  Keep paths `os.path`/`pathlib`-clean so Linux isn't broken.

## UI

- **Gradio** (Blocks API) — this is a firm choice, not a default. The whole
  product thesis is "Gradio-style WebUI, not node graph." Do not introduce a
  node/graph UI abstraction.
- Custom CSS for the glassmorphism theme (see `structure.md` and the
  patterns steering doc). No heavyweight front-end framework.

## ML / inference stack

- **PyTorch** with CUDA (owner runs an **RTX 4090, 24 GB**). Default dtype
  **bfloat16**. Expose precision as a user choice: `bf16` (fits 24 GB with
  headroom), `fp8_scaled` (~13 GB, frees VRAM for other tenants), and leave
  hooks for `nvfp4`/`gguf` later.
- **diffusers** + **transformers** for model components. Krea 2 specifically:
  - Diffusion: Krea 2 single-stream DiT checkpoint.
  - VAE: `AutoencoderKLQwenImage` — this is literally the Qwen-Image VAE, so the
    VAE adapter is shared/reusable across future Qwen-Image and Krea backends.
  - Text encoder: `Qwen3VLForConditionalGeneration`
    (`Qwen/Qwen3-VL-4B-Instruct`), multi-layer hidden-state aggregation, 512
    token max, with a baked-in prompt template (the user's text is wrapped, not
    sent raw).
- **huggingface_hub** for model downloads (streaming, resumable).
- Attention: prefer PyTorch native `scaled_dot_product_attention` as the safe
  default. Optional SageAttention/FlashAttention/xformers behind feature flags,
  installed-but-not-forced (Forge's own design lesson). Never hard-fail if an
  optional attention backend is missing — fall back down the chain.

## Persistence

- **SQLite** via Python's stdlib `sqlite3`. One DB file (`studio.db`). No ORM
  for Phase 1 — plain parameterized SQL in a thin `db/` module, mirroring the
  BeatBunny `db/db.py` shape. Artifacts (images) written to
  `outputs/job_<id>/`, paths recorded in the DB.

## Config

- **python-dotenv** `.env` driven, mirroring BeatBunny. Minimum keys:
  `MODEL_DIR`, `OUTPUT_DIR`, `APP_NAME`, `DB_PATH`. Provide `.env.example`.
- A central `Config` object resolves all paths off a single base dir. No path
  literals scattered through the code.

## Dependency discipline (this is a product feature, not hygiene)

- **Pin everything.** `requirements.txt` with exact `==` versions, generated
  from a known-good resolved set. The "still works tomorrow" promise depends on
  this. A `uv.lock` (or equivalent) is committed.
- **No auto-self-update.** The app never `git pull`s or `pip install`s itself on
  launch. Updates are a separate, deliberate, user-run action. (This is the
  exact A1111 failure mode being designed out — see Layer 1 §3.1.)
- **Decouple model backends from the shell.** A model backend importing a broken
  or missing dependency must degrade to a disabled-with-reason state in the UI,
  never crash the app. Backend imports are lazy and guarded.

## Testing

- **pytest.** The reference patterns (from BeatBunny/Higgs) were hard-won and
  test-driven; keep them that way. Every pattern module (downloader,
  system_check, db, vram_manager, generator) ships with tests. See the patterns
  steering doc for what specifically must be covered. Prefer testing behavior
  (EARS acceptance criteria map to test cases) over implementation detail.

## Explicit non-choices (do not introduce)

- No node-graph engine. No A1111/Forge codebase fork (we departed from it on
  purpose). No ORM in Phase 1. No web framework beyond Gradio. No auto-updater.
  No bundling of model weights into the repo (weights download at runtime;
  they also carry a separate license — see Layer 3 §7).

## Krea 2 canonical files (from Comfy-Org/Krea-2, the reference integration)

Use these exact filenames/roles when wiring the Krea 2 backend and downloader:

- Diffusion (Turbo, Phase 1 default): `krea2_turbo_fp8_scaled.safetensors`
  (~13 GB) or `krea2_turbo_bf16.safetensors` (~25 GB).
- Diffusion (RAW, later phases / training base): `krea2_raw_bf16.safetensors`
  — note: RAW checkpoint key layout has known ComfyUI-format quirks; treat as
  Phase 2+, not Phase 1.
- Text encoder: `qwen3vl_4b_fp8_scaled.safetensors`.
- VAE: `qwen_image_vae.safetensors`.
- Turbo sampler defaults: **8 steps, CFG 1.0 (disabled), fixed mu/shift 1.15**.
- RAW sampler defaults (later): 28–52 steps, CFG 3.5–4.5.

Full integration detail and citations live in the patterns steering doc and
`#[[file:../../spec-research/layer3-krea2-inference-interface.md]]`.
