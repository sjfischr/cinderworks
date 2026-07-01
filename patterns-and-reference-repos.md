# Patterns & Reference Repositories (READ FIRST)

> **This is the most important steering file.** The patterns below were built
> and debugged the hard way across the owner's prior projects. They were
> **test-driven and genuinely challenging to get working**. Kiro must **reuse
> these patterns, not reinvent them.** When in doubt, go read the actual source
> in the referenced repos and mirror it. Reimplementing from scratch is a
> regression, not an improvement.

---

## 1. Reference repositories (ground truth — consult before writing a module)

| Repo / source | What to take from it | URL |
| --- | --- | --- |
| **BeatBunny** (owner) | The canonical template: `app.py` shell, `worker/` split, `db/db.py`, streaming downloader generator, system_check, glassmorphism CSS, tabbed layout. This is the skeleton the Studio copies. | https://github.com/sjfischr/BeatBunny — `app.py` at https://raw.githubusercontent.com/sjfischr/BeatBunny/main/app.py |
| **Higgs Studio** (owner) | Same template applied to a real HF-transformers model; refinements: surfaced control tokens as first-class UI, correct-ported-build discipline, responsible-use framing. | https://github.com/sjfischr/higgs-studio |
| **Krea 2** (official) | Ground-truth inference code: DiT config, sampler math (euler flow + resolution-aware mu shift), VAE class, Qwen3-VL conditioner + baked prompt template. | https://github.com/krea-ai/krea-2 — inference.py, sampling.py, autoencoder.py, encoder.py under raw.githubusercontent.com/krea-ai/krea-2/main/ |
| **Comfy-Org/Krea-2** (HF) | Canonical model **filenames**, precision variants, and the reference load order the Studio's backend should mirror. | https://huggingface.co/Comfy-Org/Krea-2 |
| **ComfyUI Krea2ImageNode** | The reference native integration (v0.26.0). Load-encode-offload sequence, sampler defaults, known gotchas. | https://docs.comfy.org/built-in-nodes/Krea2ImageNode |
| **Forge Neo** (Haoming02) | Memory-management discipline to emulate; batch-control UX; attention fallback chain; model-path federation idea (later). NOT to be forked. | https://github.com/Haoming02/sd-webui-forge-classic (neo branch) |

Local research (authoritative, already distilled from the above):
- `#[[file:../../spec-research/layer1-forge-neo-and-competitive.md]]`
- `#[[file:../../spec-research/layer2-reference-patterns.md]]`
- `#[[file:../../spec-research/layer3-krea2-inference-interface.md]]`
- `#[[file:../../spec-research/layer4-gap-analysis.md]]`

---

## 2. The reusable patterns (reproduce these exactly)

Each pattern maps to a module in `structure.md` and to an incumbent pain it
solves. Each has a **test obligation** — these were test-driven originally and
must stay that way.

### Pattern A — Streaming model downloader
- **Shape:** a Python **generator** that `yield`s human-readable progress
  strings; a thin wrapper streams those to a Gradio Textbox. Resumable, uses
  `huggingface_hub`. Places files in the correct `MODEL_DIR` subfolders
  automatically. Companion fns: model-info text, download-state check, hub
  reachability check.
- **Source to mirror:** BeatBunny `worker/model_downloader.py`
  (`download_all_models_generator`, `get_model_info_text`, `get_download_state`,
  `check_huggingface_hub`).
- **Solves:** confusing manual model install (Layer 1 §3.9).
- **Test obligation:** resume-after-interruption yields correct progress;
  missing-hub is reported, not thrown; wrong/partial file is detected by the
  duck-typed presence check, not assumed present.

### Pattern B — System status checker / readiness banner
- **Shape:** pure functions `check_cuda_status`, `check_model_status`,
  `get_system_status_text`, `is_ready_to_generate`; a `get_readiness_banner`
  returns a Gradio update that shows a plain-language banner when not ready and
  hides it when ready.
- **Source to mirror:** BeatBunny `worker/system_check.py`.
- **Solves:** cryptic errors (Layer 1 §3.3), VRAM opacity (§3.7).
- **Test obligation:** every not-ready reason produces a specific, human
  sentence (no "generation failed"); ready state only true when CUDA + weights
  + VAE + encoder are all actually present.

### Pattern C — Lazy model loading
- **Shape:** weights are **not** loaded at boot. App opens instantly; the model
  loads on the first generate call, behind the readiness check.
- **Source to mirror:** BeatBunny boot flow (no model in memory until generate).
- **Solves:** slow startup (Layer 1 §3.6).
- **Test obligation:** app import + UI construction does not touch CUDA or load
  weights; first generate triggers exactly one load; second generate reuses it.

### Pattern D — SQLite persistence
- **Shape:** plain `sqlite3`, no ORM. `init_db`, `create_job`,
  `get_recent_jobs`, `get_job`, `get_job_artifacts`. Every job records prompt,
  all params, seed, model id, timing, and artifact paths. History is a
  first-class tab, not an afterthought.
- **Source to mirror:** BeatBunny `db/db.py`.
- **Solves:** the #1 unaddressed incumbent request — session persistence
  (Layer 1 §3.8; A1111 issue #842, open since 2022).
- **Test obligation:** a completed generation is fully reconstructable from the
  DB alone (re-run with identical params → identical seed handling); history
  paging does not load all rows into memory.

### Pattern E — Glassmorphism Gradio shell
- **Shape:** animated multi-stop gradient background; `.glass-panel` with
  `backdrop-filter: blur(16px)`, translucent white borders; forced light text
  for contrast. Lemon/amber accent palette consistent with the owner's other
  tools.
- **Source to mirror:** BeatBunny `app.py` `CUSTOM_CSS` block (reuse the actual
  gradient/blur values; don't approximate).
- **Solves:** dated/clunky UI (Layer 1 §3.5).
- **Test obligation:** none functional; visual. Keep CSS in `ui/theme.py`, not
  inline in `app.py`.

### Pattern F — Tabbed layout
- **Shape:** `gr.Blocks` → `gr.Tabs` → one tab per functional area. Phase 1
  tabs: **Generate**, **History**, **Models** (download), **Settings**.
- **Source to mirror:** BeatBunny/Higgs tab structure.

### Pattern G — Plain-language error surfacing
- **Shape:** every handler wrapped try/except; on failure it returns a
  `❌ <plain language>` string and hides the spinner. Tracebacks to the log
  file only.
- **Source to mirror:** BeatBunny generation handlers.
- **Solves:** red-wall failures (Layer 1 §3.3).
- **Test obligation:** a forced backend exception yields a user-facing string
  and a logged traceback, and does not propagate to Gradio.

### Pattern H — Aggressive unload / VRAM tenant management (NEW, elevated)
- **Shape:** a small coordinator all GPU consumers register with. Rule: **one
  heavyweight model resident at a time** unless the user opts into co-residency
  and has the VRAM. Even within one generation, mirror ComfyUI's Krea 2
  behavior: load text encoder → encode prompt → **offload encoder** → load
  diffusion model → sample → decode. Encoder and DiT are separate tenants and
  should not both sit at peak VRAM.
- **Why elevated:** Phase 3 (prompt LLM) and Phase 4 (trainer) make VRAM
  genuinely multi-tenant. Build the tenant interface now, even though Phase 1
  only has encoder + DiT tenants, so later phases slot in without a rewrite.
- **Source to emulate:** Forge Neo `memory_management.py` discipline (unload as
  default, not just OOM fallback) + ComfyUI Krea 2 encode-then-offload flow.
- **Solves:** VRAM opacity + future multi-tenant OOM (Layer 1 §3.7, Layer 4 §6.2).
- **Test obligation:** after encode, encoder VRAM is released before DiT loads
  (assert via allocation tracking / mock); registering a second tenant while
  one is resident triggers an unload, not an OOM.

### Higgs refinements to carry forward
- **Surface the knobs.** Sampler params (steps, cfg, mu/shift, seed, size,
  batch) are real UI controls with honest tooltips — not hidden flags. Exception:
  Krea's baked-in prompt template is intentionally not user-editable; document
  it, don't expose it.
- **Correct-ported-build discipline.** Always target the verified
  inference-compatible checkpoint/build (the Comfy-Org filenames above), never
  raw upstream weights that need a different serving stack.

---

## 3. Krea 2 integration reference (mirror this, don't guess)

Verified from Krea's own source + the ComfyUI native integration.

**Components & canonical files** (from Comfy-Org/Krea-2):
- Diffusion DiT (Phase 1 = Turbo): `krea2_turbo_fp8_scaled.safetensors` (~13 GB)
  or `krea2_turbo_bf16.safetensors` (~25 GB).
- Text encoder: `qwen3vl_4b_fp8_scaled.safetensors` (Qwen3-VL-4B-Instruct).
- VAE: `qwen_image_vae.safetensors` (`AutoencoderKLQwenImage`, f8 / 16 latent ch).

**Load / generate sequence** (the Pattern H flow, concretely):
1. Load Qwen3-VL-4B text encoder to GPU.
2. Wrap the user prompt in Krea's baked system/user template; encode; pull
   hidden states from the 12 selected layers (multi-layer aggregation).
3. **Offload the text encoder** (release VRAM).
4. Load the Krea 2 Turbo DiT.
5. Sample: euler integration of the flow-matching ODE. Turbo uses **8 steps,
   CFG disabled (1.0), fixed mu/shift 1.15**. (RAW, later: 28–52 steps, CFG
   3.5–4.5, resolution-aware mu interpolated between y1=0.5 @256 and y2=1.15
   @1280.) Batch: image *i* uses `seed + i`.
6. Decode latents with the Qwen-Image VAE (tiled decode available for headroom).

**DiT config (for reference):** single-stream DiT, features 6144, 28 layers,
48 heads, patch 2, 16 channels, bf16.

**Known gotchas (design around these):**
- **RAW is not plug-and-play** — its checkpoint key layout differs from the
  ComfyUI/expected format. Reinforces Turbo-first for Phase 1.
- **Apple MPS black images** — needs fp16 VAE; not our primary platform but keep
  the VAE dtype configurable.
- **Quantized weights can anchor in VRAM** after use (bitsandbytes reports);
  the vram_manager must actually free, and tests must verify it.
- **"No checkpoint found"** class errors when files are misplaced — the
  downloader's auto-placement + the system_check duck-typed presence check are
  the defense; surface a plain-language "model not downloaded yet" banner.

**Licensing reminder:** Krea 2 **code** is Apache-2.0; the **weights** are under
Krea's separate community license. Download weights at runtime; never vendor
them into the repo. Fine for personal use; matters if the tool is ever
distributed commercially. See Layer 3 §7.

---

## 4. How Kiro should use this doc

- Before implementing any module named in `structure.md`, open the matching
  "Source to mirror" and follow its shape.
- Treat each **test obligation** as a required test in `tests/`.
- If a requirement in the spec seems to conflict with a pattern here, stop and
  flag it rather than silently diverging — the patterns encode debugging that
  already happened.
