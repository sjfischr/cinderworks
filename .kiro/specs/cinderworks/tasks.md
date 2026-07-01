# Implementation Plan: Cinderworks Phase 1

## Overview

Implement the core creative loop for Cinderworks: a local-first Gradio image generation studio with one-click install, lazy model loading, streaming downloads, VRAM tenant discipline, image generation via Krea 2 Turbo, and persistent history. The implementation follows the BeatBunny/Higgs Studio template structure and mirrors its proven patterns.

## Tasks

- [x] 1. Set up project structure, config, and bootstrap
  - [x] 1.1 Create directory structure and config module
    - Create `studio/` root with all subdirectories: `ui/`, `core/`, `models/`, `models/backends/`, `db/`, `install/`, `outputs/`, `models_store/`, `tests/`
    - Create `studio/config.py` with a `Config` object that resolves `MODEL_DIR`, `OUTPUT_DIR`, `APP_NAME`, `DB_PATH` from `.env` using `python-dotenv`
    - Create `studio/.env.example` with all required keys
    - Use `pathlib` for all path handling (platform-agnostic)
    - _Requirements: 1.1, 14.1_

  - [x] 1.2 Create bootstrap scripts
    - Create `studio/install/bootstrap.bat` (Windows) and `studio/install/bootstrap.sh` (Linux)
    - Scripts must: check for Python 3.11, Git, and uv; create project-local venv via uv; install exact-pinned deps from lock file; launch the Gradio server
    - If a prerequisite is missing, report which one by name and exit non-zero without creating the venv
    - Scripts must NOT perform git pull or self-directed pip install
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.3 Create pinned requirements.txt
    - Define exact `==` pinned versions for all dependencies: gradio, torch, diffusers, transformers, huggingface_hub, python-dotenv, safetensors, accelerate, hypothesis, pytest
    - _Requirements: 1.1_

- [x] 2. Implement database layer
  - [x] 2.1 Implement `db/db.py` with schema and CRUD operations
    - Create `job` and `artifact` tables (schema per design doc)
    - Implement `init_db()`, `create_job()`, `get_recent_jobs(limit=20, offset=0)`, `get_job(job_id)`, `get_job_artifacts(job_id)`
    - Use plain `sqlite3` with parameterized SQL, no ORM
    - `get_recent_jobs` must return results ordered by `created_at DESC` with proper paging (LIMIT/OFFSET, not SELECT all)
    - Store params as JSON blob in `params_json`; artifact metadata (path, seed, width, height) in artifact table
    - _Requirements: 8.1, 8.3, 8.5_

  - [x] 2.2 Write property test for job persistence round-trip
    - **Property 15: Job persistence round-trip**
    - **Validates: Requirements 8.1**

  - [x] 2.3 Write property test for history listing order
    - **Property 16: History listing ordered by creation time descending**
    - **Validates: Requirements 8.3**

  - [x] 2.4 Write property test for history paging
    - **Property 18: History paging returns correct page sizes**
    - **Validates: Requirements 8.5**

  - [x] 2.5 Write unit tests for db module
    - Test DB write failure handling (surfaced error, image retained)
    - Test missing image file → placeholder behavior
    - Test `get_recent_jobs` returns prompts truncated to 120 characters
    - _Requirements: 8.2, 8.6_

- [x] 3. Implement VRAM manager
  - [x] 3.1 Implement `core/vram_manager.py` with tenant acquire/release API
    - Implement `VRAMManager` class with `acquire(tenant)`, `release(tenant)`, `estimate_available()`, `can_fit(bytes_needed)` methods
    - Enforce one heavyweight tenant resident at a time: acquiring tenant B while tenant A is resident unloads A first
    - `acquire` failure due to insufficient memory → raise with plain-language OOM message
    - Precision-aware: update available-memory estimate based on precision (bf16 vs fp8_scaled)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 3.2 Write property test for tenant discipline
    - **Property 14: Tenant discipline — acquire unloads existing resident**
    - **Validates: Requirements 7.2, 7.3**

  - [x] 3.3 Write property test for VRAM batch refusal
    - **Property 12: VRAM manager refuses batch exceeding estimated capacity**
    - **Validates: Requirements 6.4**

  - [x] 3.4 Write property test for OOM preservation
    - **Property 13: OOM during batch preserves prior completed batches**
    - **Validates: Requirements 6.5**

  - [x] 3.5 Write unit tests for VRAM manager
    - Test encoder released before DiT (mock allocation tracking)
    - Test acquire failure returns plain-language OOM message suggesting lower batch size or fp8_scaled
    - _Requirements: 7.2, 7.5_

- [x] 4. Implement system check and readiness
  - [x] 4.1 Implement `core/system_check.py`
    - Implement `check_cuda_status() → bool` (detects CUDA without triggering model loads)
    - Implement `check_model_status() → dict[str, bool]` (presence + size check; partial file = not present)
    - Implement `is_ready_to_generate() → bool` (CUDA + all three model files present and valid)
    - Implement `get_system_status_text() → str` (plain-language summary of all conditions)
    - Implement `get_readiness_banner() → GradioUpdate` (shows/hides banner with all unmet conditions)
    - On startup: detect CUDA, store result in readiness state, continue regardless of outcome
    - _Requirements: 1.5, 3.6, 4.1, 4.2, 4.3_

  - [x] 4.2 Write property test for readiness reporting
    - **Property 6: Readiness reports all unmet conditions**
    - **Validates: Requirements 4.1, 4.4**

  - [x] 4.3 Write property test for size mismatch detection
    - **Property 3: Size mismatch means not-present**
    - **Validates: Requirements 3.6**

  - [x] 4.4 Write unit tests for system check
    - Test ready only when CUDA + all 3 files present and size-valid
    - Test every not-ready reason is a specific sentence
    - Test CUDA absence stored in readiness state; app continues startup
    - _Requirements: 1.5, 4.1_

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement model registry and downloader
  - [x] 6.1 Implement `models/registry.py`
    - Define `RegistryEntry` dataclass with model_id, display_name, backend_module, checkpoints, vae, text_encoder, sampler_defaults, precision_options, vram_tiers
    - Implement `list_models()`, `get_meta(model_id)`, `run_generation(model_id, params)` as public API
    - Lazy import of backend modules: import only on first access, guarded with try/except
    - If import fails: mark backend unavailable-with-reason, do not re-attempt
    - If generation requested against unavailable backend: refuse with recorded reason
    - Ship exactly one registry entry: Krea 2 Turbo (no stubs)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 6.2 Write property test for backend unavailability round-trip
    - **Property 19: Backend unavailability reason round-trip**
    - **Validates: Requirements 9.3, 9.4**

  - [x] 6.3 Write unit tests for registry
    - Test shell never imports backend directly (all access through registry)
    - Test failing backend is marked unavailable and app still constructs
    - Test exactly 1 entry in Phase 1
    - _Requirements: 9.1, 9.3, 9.6_

  - [x] 6.4 Implement `models/downloader.py`
    - Implement `download_all_models_generator(model_id) → Generator[str]` yielding progress per chunk
    - Implement `get_model_info_text(model_id) → str` and `get_download_state(model_id) → dict[str, str]`
    - Implement `check_huggingface_hub() → bool` (reachability check with 30s timeout)
    - Use `huggingface_hub` for streaming, resumable downloads; auto-place files into `MODEL_DIR` subfolders
    - Progress strings must contain: current filename, percentage, bytes downloaded vs total
    - Resume from last byte on re-trigger (not from zero)
    - Already-present files (passing size check) → report "already downloaded"
    - Partial failure: report which files failed, retain successful ones, allow re-trigger for failed only
    - Hub unreachable: report in plain language, no crash, UI stays responsive
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8_

  - [x] 6.5 Write property tests for downloader
    - **Property 1: Download progress contains required information**
    - **Property 2: Download resumes from interruption point**
    - **Property 4: Partial download failure identifies exactly the failed files**
    - **Property 5: Download yields progress at least once per chunk**
    - **Validates: Requirements 3.1, 3.2, 3.7, 3.8**

  - [x] 6.6 Write unit tests for downloader
    - Test hub unreachable returns string not exception
    - Test already-present detection (size matches → skip)
    - Test auto-placement of 3 files into correct subdirectories
    - _Requirements: 3.4, 3.5, 3.3_

- [x] 7. Implement Krea 2 Turbo backend
  - [x] 7.1 Implement `models/backends/krea2.py`
    - Implement load→encode→offload→load→sample→decode sequence
    - Text encoder: load Qwen3-VL-4B, wrap prompt in baked template, encode with multi-layer hidden-state aggregation (12 layers), offload via `vram_manager.release()`
    - Diffusion: load Krea 2 Turbo DiT at chosen precision via `vram_manager.acquire()`, Euler flow sampling with Turbo_Defaults (8 steps, CFG 1.0 disabled, mu/shift 1.15)
    - VAE: decode latents with Qwen-Image VAE (tiled decode option)
    - Batch: image *i* uses `base_seed + i`; total images = batch_size × batch_count
    - All GPU moves go through `vram_manager` (never call `.to('cuda')` directly)
    - Yield progress updates: encoding started, sampling step N of M, decoding
    - Empty prompt → refuse with plain-language message
    - Return images + resolved params (including actual seed) for persistence
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 6.1, 6.2, 7.1, 7.2_

  - [x] 7.2 Write property tests for Krea 2 backend
    - **Property 8: Sampler parameters default to Turbo unless explicitly overridden**
    - **Property 9: Seed determinism**
    - **Property 10: Parameter bounds validation**
    - **Property 11: Batch produces correct image count with correct per-image seeds**
    - **Validates: Requirements 5.2, 5.4, 5.5, 6.1, 6.2**

  - [x] 7.3 Write unit tests for Krea 2 backend
    - Test encode→offload→sample→decode order via mocked harness
    - Test Turbo defaults applied when params omitted
    - Test empty prompt refused with plain-language message
    - Test batch_size > 1 produces correct number of images with sequential seeds
    - _Requirements: 5.1, 5.7, 6.1_

- [x] 8. Implement model loader (lazy loading)
  - [x] 8.1 Implement `core/model_loader.py`
    - Load model components only on first generate (not on boot)
    - Cache loaded components keyed by `(model_id, precision)`
    - Importing this module must NOT touch CUDA or trigger weight loading
    - Delegate GPU placement to `vram_manager`
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 8.2 Write unit tests for model loader
    - Test importing module does not trigger CUDA initialization
    - Test first generate loads model, subsequent reuses cached
    - Test no model weights in GPU memory before first generate
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 9. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement UI layer
  - [x] 10.1 Implement `ui/theme.py` with glassmorphism CSS
    - Animated multi-stop gradient background
    - `.glass-panel` with `backdrop-filter: blur(16px)`, translucent borders
    - Forced light text for contrast; lemon/amber accent palette
    - Source from BeatBunny `CUSTOM_CSS` block
    - _Requirements: 10.1 (visual, non-functional)_

  - [x] 10.2 Implement `ui/controls.py` with parameter controls
    - Prompt textbox (no internal template exposed)
    - Sampler params: steps (1–100, default 8), seed (0–2³²-1), width/height (512–2048, multiples of 64, default 1024)
    - Precision picker: bf16 / fp8_scaled
    - Batch_Size (1–16, default 1) and Batch_Count (1–100, default 1) as distinct controls
    - Tooltips on all controls explaining their effect (batch_size=simultaneous/VRAM, batch_count=sequential/queue)
    - Validate all parameter bounds before passing to generation
    - _Requirements: 5.5, 5.6, 6.3_

  - [x] 10.3 Implement `ui/handlers.py` with error boundary and event handlers
    - Implement `on_generate()`: check readiness → validate params → run generation → persist job → display results
    - Implement `on_download()`: trigger downloader generator, stream progress to UI
    - Implement `on_load_history()`: paginated job listing (20 per page)
    - Implement `on_load_params()`: populate Generate tab fields from past Job without triggering generation
    - All handlers wrapped in try/except: catch, log traceback to file, return `❌ <plain language>` string
    - `friendly()` error mapper: known classes → specific messages + log path; unknown → generic + log path
    - Never render raw tracebacks or exception class names in UI
    - While generating: yield progress updates (encoding, step N of M, decoding)
    - If not ready: display inline `❌` error with specific unmet condition(s), do not initiate generation
    - _Requirements: 4.4, 4.5, 5.8, 8.3, 8.4, 8.5, 11.1, 11.3_

  - [x] 10.4 Write property tests for handlers
    - **Property 7: Error handler produces plain-language output without tracebacks**
    - **Property 20: Graceful degradation preserves app state on failure**
    - **Validates: Requirements 4.5, 11.1, 11.3**

  - [x] 10.5 Write property test for parameter reload
    - **Property 17: Job params reload populates generation fields exactly**
    - **Validates: Requirements 8.4**

  - [x] 10.6 Write unit tests for handlers
    - Test spinner on/off lifecycle
    - Test empty prompt refused before generation starts
    - Test handler never re-raises to Gradio (always returns string)
    - Test generation refused when not ready, with specific unmet conditions listed
    - _Requirements: 5.7, 4.4, 11.1_

- [x] 11. Implement app shell and wire everything together
  - [x] 11.1 Implement `app.py` — Gradio Blocks shell
    - Build `gr.Blocks` with theme from `ui/theme.py`
    - Create four tabs: Generate, History, Models, Settings
    - Wire component events to `ui/handlers.py` functions
    - On load: call `core.system_check.get_readiness_banner()` for initial banner state
    - Shell stays thin: no inference, no download, no SQL in app.py
    - Shell never imports backends directly (all through registry)
    - Readiness banner hides within 2 seconds when ready (reactive update on download complete)
    - Import of model-loading libraries deferred (no CUDA init on startup)
    - UI must be fully navigable within 5 seconds (no model loading at boot)
    - _Requirements: 1.4, 2.1, 2.4, 4.2, 9.1, 10.1_

  - [x] 11.2 Wire Generate tab
    - Connect prompt, sampler controls, batch controls, precision picker, and generate button to `on_generate` handler
    - Connect progress output and image gallery display
    - Wire readiness banner visibility to system_check state
    - _Requirements: 5.6, 5.8, 6.3_

  - [x] 11.3 Wire History tab
    - Connect history list (paginated, 20 per page) to `on_load_history`
    - Display: truncated prompt (120 chars), thumbnail, seed, steps, resolution, precision, model_id
    - "Load more" control for next page
    - Connect "load params" action to `on_load_params` (fills Generate tab fields without generating)
    - Missing image → placeholder displayed (not crash)
    - _Requirements: 8.3, 8.4, 8.5, 8.6_

  - [x] 11.4 Wire Models tab
    - Connect download trigger to `on_download` handler
    - Display per-file progress (filename, percentage, bytes downloaded/total)
    - Show "already downloaded" status for present files
    - Download completion → re-evaluate readiness → update banner
    - _Requirements: 3.1, 3.5, 4.3_

- [x] 12. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties defined in the design document
- Unit tests validate specific examples and edge cases
- All code uses Python 3.11, pytest for testing, and hypothesis for property-based tests
- The BeatBunny/Higgs Studio patterns document is the authoritative reference for module implementation details
- All paths must use `pathlib` for cross-platform compatibility (Requirements 14.1, 14.2)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "3.2", "3.3", "3.4", "3.5", "4.2", "4.3", "4.4"] },
    { "id": 3, "tasks": ["6.1", "6.4"] },
    { "id": 4, "tasks": ["6.2", "6.3", "6.5", "6.6", "7.1", "8.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "8.2"] },
    { "id": 6, "tasks": ["10.1", "10.2"] },
    { "id": 7, "tasks": ["10.3"] },
    { "id": 8, "tasks": ["10.4", "10.5", "10.6"] },
    { "id": 9, "tasks": ["11.1"] },
    { "id": 10, "tasks": ["11.2", "11.3", "11.4"] }
  ]
}
```
