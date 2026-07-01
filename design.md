# Design — Image Studio Core (Phase 1)

> **Spec:** `image-studio-core` · **Phase:** 1. Implements
> `#[[file:requirements.md]]`. Governed by
> `#[[file:../../steering/product.md]]`, `#[[file:../../steering/tech.md]]`,
> `#[[file:../../steering/structure.md]]`, and — most importantly —
> `#[[file:../../steering/patterns-and-reference-repos.md]]`, which contains the
> hard-won, test-driven patterns this design reuses rather than reinvents.
>
> Design rule: where this document and the patterns steering doc describe the
> same thing, the patterns doc's "source to mirror" wins. Go read the BeatBunny
> source before implementing the corresponding module.

## 1. Architecture overview

A single-process Gradio app with a thin shell and clear module seams. Layers:

```
┌─────────────────────────────────────────────────────────────┐
│  app.py  (Gradio Blocks shell — tabs, wiring only)          │
│  Tabs: Generate │ History │ Models │ Settings               │
└───────────────┬─────────────────────────────────────────────┘
                │  calls
        ┌───────▼────────┐   ui/handlers.py  (try/except → plain text)
        │  ui/           │   ui/controls.py  (param + batch controls)
        │  handlers,     │   ui/theme.py     (glassmorphism CSS)
        │  controls,     │
        │  theme         │
        └───────┬────────┘
                │  delegates to
   ┌────────────▼────────────────────────────────────────────┐
   │  core/                models/                 db/        │
   │  system_check    ┌─►  registry ──► backends/   db.py     │
   │  model_loader    │    downloader     krea2.py            │
   │  vram_manager ◄──┘                    │                  │
   │      ▲───────────────────────────────┘ (all GPU moves)  │
   └─────────────────────────────────────────────────────────┘
```

Key invariants (from `structure.md`, restated because they carry the design):
- The shell never imports `backends/krea2.py` directly — only via `registry`.
- Only `vram_manager` moves anything on/off the GPU.
- Only `db.py` touches SQLite.
- The UI boundary never sees a traceback.

## 2. Components

### 2.1 `app.py` — shell (R1, R6, F: tabbed layout)
Builds `gr.Blocks` with the glass theme from `ui/theme.py` and four tabs. Wires
component events to functions in `ui/handlers.py`. Holds no logic. On load, it
calls `core.system_check.get_readiness_banner()` to set the initial banner
state.

### 2.2 `ui/handlers.py` — handlers & error boundary (R4, R5, R8)
One handler per user action (download, generate, open-history, load-params).
Each is wrapped:

```python
def on_generate(...):
    try:
        yield spinner_on()
        for update in run_generation(...):   # delegates to models/registry
            yield update
    except Exception as e:
        log.exception("generate failed")
        yield error_text(f"❌ {friendly(e)}"), spinner_off()
```

`friendly()` maps known failure classes (OOM, missing file, hub unreachable) to
plain sentences; unknown errors get a generic "something went wrong — see log"
plus the log path. Never re-raises to Gradio.

### 2.3 `ui/controls.py` — parameter surface (R5, R6)
Reusable control groups: prompt box; sampler params (steps, seed, width,
height) with Turbo defaults pre-filled; **precision picker** (bf16 / fp8_scaled);
**batch size** and **batch count** as two clearly separated controls with the
size-vs-count tooltip (R6). Krea's prompt template is NOT surfaced (R5).

### 2.4 `core/system_check.py` — readiness (R1, R3, R4)
Mirrors BeatBunny `worker/system_check.py`. Pure functions:
`check_cuda_status()`, `check_model_status()` (duck-typed presence + size
sanity, so a partial file reads as not-present per R3), `is_ready_to_generate()`,
`get_system_status_text()`, `get_readiness_banner()` (returns a Gradio update).

### 2.5 `core/model_loader.py` — lazy loading (R2)
Loads model components only on first generate. Caches loaded components keyed by
(model_id, precision). Delegates actual GPU placement to `vram_manager`. Import
of this module must not touch CUDA.

### 2.6 `core/vram_manager.py` — tenant discipline (R7, Pattern H)
The coordinator. Tenants register; it enforces "one heavyweight tenant resident
at a time" for Phase 1 and exposes `acquire(tenant)` / `release(tenant)`.
Emulates Forge's unload-as-default discipline. This is the single chokepoint for
`.to('cuda')` / `.to('cpu')`. Built now with a real interface even though Phase 1
only has two tenants (encoder, DiT), so Phase 3/4 tenants slot in unchanged.

### 2.7 `models/registry.py` — model-agnostic seam (R9)
Holds registry entries and resolves a `model_id` to its backend module, metadata,
and defaults. Public surface the shell uses: `list_models()`, `get_meta(id)`,
`run_generation(id, params)` (generator yielding progress + final image).
Backend import is lazy and guarded: a failing backend is recorded as
unavailable-with-reason (R9) and does not propagate. **Phase 1: exactly one
entry, Krea 2 Turbo. No stubs.**

### 2.8 `models/downloader.py` — streaming downloader (R3, Pattern A)
Mirrors BeatBunny `worker/model_downloader.py`. A generator yielding progress
strings; resumable via `huggingface_hub`; auto-places the three Krea files
(`krea2_turbo_*`, `qwen3vl_4b_fp8_scaled`, `qwen_image_vae`) into the correct
`MODEL_DIR` subfolders; reports hub-unreachable without throwing; detects
already-present and partial files.

### 2.9 `models/backends/krea2.py` — Krea 2 Turbo backend (R5, R7)
Implements the load→encode→offload→load→sample→decode sequence from the patterns
doc §3, using `vram_manager` for every GPU move:

1. `vram_manager.acquire(text_encoder)`; load Qwen3-VL-4B.
2. Wrap prompt in Krea's baked template; encode; aggregate the 12 selected
   hidden-state layers.
3. `vram_manager.release(text_encoder)`.
4. `vram_manager.acquire(dit)`; load Krea 2 Turbo DiT at chosen precision.
5. Euler flow sampling: 8 steps, CFG off, mu/shift 1.15 (Turbo). Batch image *i*
   uses `seed + i`.
6. Decode via Qwen-Image VAE (tiled decode option for headroom).
Returns image(s) + the resolved params (including the actual seed used) for
persistence.

### 2.10 `db/db.py` — persistence (R8, Pattern D)
Plain `sqlite3`. Mirrors BeatBunny `db/db.py`. Functions: `init_db`,
`create_job`, `get_recent_jobs(limit, offset)` (paged, R8), `get_job`,
`get_job_artifacts`.

## 3. Data model

```sql
CREATE TABLE job (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at    TEXT NOT NULL,          -- ISO8601
  model_id      TEXT NOT NULL,          -- 'krea2-turbo'
  prompt        TEXT NOT NULL,
  params_json   TEXT NOT NULL,          -- steps, cfg, mu, width, height,
                                        --   precision, batch_size, batch_count
  seed          INTEGER NOT NULL,       -- the actual seed used (R5)
  duration_ms   INTEGER,
  status        TEXT NOT NULL           -- 'complete' | 'failed'
);

CREATE TABLE artifact (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id        INTEGER NOT NULL REFERENCES job(id),
  path          TEXT NOT NULL,          -- outputs/job_<id>/<n>.png
  seed          INTEGER NOT NULL,       -- per-image seed (seed + i)
  width         INTEGER,
  height        INTEGER
);
```

A job is fully reconstructable from `job` alone (R8 reproducibility): re-running
`prompt` + `params_json` + `seed` reproduces the run.

## 4. Key sequence — a generation (traces R2, R4, R5, R7, R8)

```
User clicks Generate
  → handlers.on_generate (spinner on)
    → system_check.is_ready_to_generate?
        no  → yield "❌ <specific reason>" ; stop            (R4)
        yes → registry.run_generation('krea2-turbo', params) (R9)
                → krea2 backend:
                    vram_manager.acquire(encoder)             (R7)
                    load encoder (first use only)             (R2)
                    encode(prompt via baked template)         (R5)
                    vram_manager.release(encoder)             (R7, Pattern H)
                    vram_manager.acquire(dit)
                    load DiT @ precision                      (R2)
                    sample 8-step euler, seed+i per image     (R5)
                    decode via Qwen VAE
                    → yield progress strings throughout       (Pattern G)
              → write images to outputs/job_<id>/             (R8)
              → db.create_job(...) + artifacts                (R8)
              → yield final image(s) ; spinner off
```

Any exception inside is caught at the handler boundary → plain text + logged
traceback (R4).

## 5. Error handling strategy (R4)

- **Not ready:** pre-checked; specific banner; generate refused with reason.
- **Hub unreachable (download):** reported string, no throw (R3).
- **OOM:** `vram_manager` attempts unload+retry once; if still OOM, plain
  "not enough VRAM for these settings" with a hint to lower batch size or switch
  to fp8 (R6/R7).
- **Missing/partial file:** treated as not-ready by `check_model_status`; banner
  says "model not downloaded yet" (R3/R4).
- **Backend import/load failure:** registry marks backend unavailable-with-reason;
  app keeps running (R9).
- All: traceback to log file, never to UI.

## 6. Technology choices (from `tech.md`, restated for traceability)

Python 3.11 + `uv` venv; Gradio Blocks; PyTorch/CUDA, default bf16 with
fp8_scaled option; diffusers + transformers for Krea components; `huggingface_hub`
for downloads; stdlib `sqlite3`; `python-dotenv`; native SDPA attention with
optional accelerated backends behind flags. No ORM, no node graph, no
auto-updater, no vendored weights.

## 7. Testing strategy (R: testability; patterns doc "test obligations")

`pytest`, one test module per component. Required behaviors:

- **downloader:** resume yields correct progress; hub-unreachable reported not
  thrown; partial file detected as not-present.
- **system_check:** every not-ready reason yields a specific sentence; ready only
  when CUDA + all three files present and size-sane.
- **model_loader:** importing the module touches neither CUDA nor weights; first
  generate loads once; second reuses.
- **vram_manager:** encoder VRAM released before DiT acquired (assert via mock
  allocation tracking); second tenant acquire while one resident triggers
  unload, not OOM.
- **registry:** shell path never imports the backend module directly; a backend
  raising on import is surfaced as unavailable, app-object still constructs.
- **db:** completed job round-trips; params reload reproduces seed handling;
  `get_recent_jobs` pages (does not select all).
- **krea2 backend:** load/encode/offload/sample/decode order verified via a
  mocked component harness (no real 13 GB download in unit tests); Turbo defaults
  applied when params omitted.

Integration smoke test (manual, gated): real download of Krea 2 Turbo fp8 → one
generation → job appears in History → params reload → re-generate reproduces.
This smoke test is also the Phase 1 success gate from `product.md`.

## 8. What this design deliberately does not do

No prompt-LLM hooks, no training hooks, no second backend, no RAW support, no
img2img/ControlNet/video, no auto-update. The seams (registry, vram_manager
tenant interface) are built to accept those later without rework, but Phase 1
ships none of them. Adding them early would violate the phasing gate and risk
the "still works tomorrow" promise before the core loop is proven used.

## 9. Open design decisions (mirror requirements' open items)

1. Default precision fp8_scaled vs bf16 (see requirements R-open-1). Design
   supports both via the precision picker; only the default value is open.
2. History delete/export deferred vs Phase 1 (requirements R-open-2). Design
   leaves room (artifact table + paths) but does not implement delete/export in
   Phase 1 unless the owner asks.
