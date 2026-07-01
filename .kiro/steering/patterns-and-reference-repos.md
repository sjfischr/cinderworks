---
inclusion: auto
---

# Patterns & Reference Repositories (READ FIRST)

> **This is the most important steering file.** The patterns below were built
> and debugged the hard way across the owner's prior projects. They were
> **test-driven and genuinely challenging to get working**. Kiro must **reuse
> these patterns, not reinvent them.** When in doubt, go read the actual source
> in the referenced repos and mirror it.

---

## 1. Reference Repositories (Ground Truth)

| Repo / Source | What to Take | URL |
| --- | --- | --- |
| **BeatBunny** (owner) | Canonical template: `app.py` shell, `worker/` split, `db/db.py`, streaming downloader generator, system_check, glassmorphism CSS, tabbed layout. | https://github.com/sjfischr/BeatBunny |
| **Higgs Studio** (owner) | Same template with HF-transformers model; surfaced control tokens as first-class UI. | https://github.com/sjfischr/higgs-studio |
| **Krea 2** (official) | Ground-truth inference: DiT config, sampler math (euler flow + mu shift), VAE class, Qwen3-VL conditioner + baked prompt template. | https://github.com/krea-ai/krea-2 |
| **Comfy-Org/Krea-2** (HF) | Canonical model filenames, precision variants, reference load order. | https://huggingface.co/Comfy-Org/Krea-2 |
| **ComfyUI Krea2ImageNode** | Reference native integration (v0.26.0). Load-encode-offload sequence, sampler defaults, known gotchas. | https://docs.comfy.org/built-in-nodes/Krea2ImageNode |
| **Forge Neo** (Haoming02) | Memory-management discipline; batch-control UX; attention fallback chain. NOT to be forked. | https://github.com/Haoming02/sd-webui-forge-classic |

---

## 2. Reusable Patterns (Reproduce Exactly)

Each pattern maps to a module in the structure steering doc and has a
**test obligation** that must be present in `tests/`.

### Pattern A — Streaming Model Downloader

- **Shape:** Python generator yielding human-readable progress strings;
  resumable via `huggingface_hub`. Auto-places files in correct `MODEL_DIR`
  subfolders. Companion fns: model-info text, download-state check, hub
  reachability check.
- **Source to mirror:** BeatBunny `worker/model_downloader.py`.
- **Solves:** confusing manual model install.
- **Test obligation:** resume-after-interruption yields correct progress;
  missing-hub is reported, not thrown; wrong/partial file detected by
  duck-typed presence check.

### Pattern B — System Status Checker / Readiness Banner

- **Shape:** Pure functions: `check_cuda_status`, `check_model_status`,
  `get_system_status_text`, `is_ready_to_generate`; a `get_readiness_banner`
  returns a Gradio update.
- **Source to mirror:** BeatBunny `worker/system_check.py`.
- **Solves:** cryptic errors, VRAM opacity.
- **Test obligation:** every not-ready reason produces a specific human
  sentence; ready state only true when CUDA + all three files present and
  size-sane.

### Pattern C — Lazy Model Loading

- **Shape:** Weights NOT loaded at boot. App opens instantly; model loads on
  first generate, behind readiness check.
- **Source to mirror:** BeatBunny boot flow.
- **Solves:** slow startup.
- **Test obligation:** app import + UI construction does not touch CUDA or load
  weights; first generate triggers exactly one load; second reuses.

### Pattern D — SQLite Persistence

- **Shape:** Plain `sqlite3`, no ORM. `init_db`, `create_job`,
  `get_recent_jobs`, `get_job`, `get_job_artifacts`. History is a first-class
  tab.
- **Source to mirror:** BeatBunny `db/db.py`.
- **Solves:** the #1 unaddressed incumbent request — session persistence.
- **Test obligation:** completed generation round-trips from DB; history paging
  does not load all rows.

### Pattern E — Glassmorphism Gradio Shell

- **Shape:** Animated multi-stop gradient background; `.glass-panel` with
  `backdrop-filter: blur(16px)`, translucent white borders; lemon/amber accent.
- **Source to mirror:** BeatBunny `app.py` `CUSTOM_CSS` block (reuse exact
  gradient/blur values).
- **Test obligation:** None functional; visual. Keep CSS in `ui/theme.py`.

### Pattern F — Tabbed Layout

- **Shape:** `gr.Blocks` → `gr.Tabs` → one tab per area. Phase 1 tabs:
  **Generate**, **History**, **Models** (download), **Settings**.
- **Source to mirror:** BeatBunny/Higgs tab structure.

### Pattern G — Plain-Language Error Surfacing

- **Shape:** Every handler wrapped try/except; on failure returns
  `❌ <plain language>` string and hides spinner. Tracebacks to log only.
- **Source to mirror:** BeatBunny generation handlers.
- **Solves:** red-wall failures.
- **Test obligation:** forced backend exception yields user-facing string and
  logged traceback, does not propagate to Gradio.

### Pattern H — Aggressive Unload / VRAM Tenant Management

- **Shape:** Small coordinator all GPU consumers register with. Rule: one
  heavyweight model resident at a time. Within one generation: load encoder →
  encode → offload encoder → load DiT → sample → decode.
- **Why elevated:** Phase 3/4 make VRAM multi-tenant. Build the interface now.
- **Source to emulate:** Forge Neo `memory_management.py` + ComfyUI Krea 2
  encode-then-offload flow.
- **Solves:** VRAM opacity + future multi-tenant OOM.
- **Test obligation:** after encode, encoder VRAM released before DiT loads;
  second tenant acquire while one resident triggers unload, not OOM.

### Higgs Refinements to Carry Forward

- **Surface the knobs.** Sampler params are real UI controls with honest
  tooltips. Exception: Krea's baked prompt template is not user-editable.
- **Correct-ported-build discipline.** Target verified inference-compatible
  checkpoints (Comfy-Org filenames), not raw upstream weights.

---

## 3. Krea 2 Integration Reference

**Components & Canonical Files** (Comfy-Org/Krea-2):
- Diffusion DiT (Turbo): `krea2_turbo_fp8_scaled.safetensors` (~13 GB) or
  `krea2_turbo_bf16.safetensors` (~25 GB).
- Text encoder: `qwen3vl_4b_fp8_scaled.safetensors` (Qwen3-VL-4B-Instruct).
- VAE: `qwen_image_vae.safetensors` (`AutoencoderKLQwenImage`, f8 / 16 ch).

**Load / Generate Sequence** (Pattern H, concretely):
1. Load Qwen3-VL-4B text encoder to GPU.
2. Wrap prompt in baked template; encode; pull hidden states from 12 selected
   layers (multi-layer aggregation).
3. **Offload text encoder** (release VRAM).
4. Load Krea 2 Turbo DiT.
5. Sample: euler flow-matching ODE. Turbo: 8 steps, CFG disabled (1.0),
   mu/shift 1.15. Batch: image *i* uses `seed + i`.
6. Decode latents with Qwen-Image VAE (tiled decode available for headroom).

**DiT config:** single-stream DiT, 6144 features, 28 layers, 48 heads, patch 2,
16 channels, bf16.

**Known Gotchas:**
- RAW checkpoint key layout differs from ComfyUI/expected → Turbo-first.
- Apple MPS produces black images → needs fp16 VAE; not primary platform.
- Quantized weights can anchor in VRAM → vram_manager must actually free.
- "No checkpoint found" errors when files misplaced → downloader auto-placement
  + system_check presence check are the defense.

**Licensing:** Krea 2 code is Apache-2.0; weights under Krea community license.
Download at runtime; never vendor into repo.

---

## 4. How to Use This Doc

- Before implementing any module, open the matching "Source to mirror" and
  follow its shape.
- Treat each test obligation as a required test in `tests/`.
- If a requirement seems to conflict with a pattern here, stop and flag it —
  the patterns encode debugging that already happened.
