# Requirements — Image Studio Core (Phase 1)

> **Spec:** `image-studio-core` · **Phase:** 1 (the first rung) · **Workflow:**
> Requirements-First (EARS).
> **Scope guard:** download Krea 2 Turbo → generate → persist → recall. Nothing
> else. Prompt-optimizer LLM, LoRA training, and additional models are later
> specs — see `steering/product.md` phasing. If an acceptance criterion here
> implies one of those, it's out of scope and belongs in that phase.
>
> Steering context that governs this spec:
> `#[[file:../../steering/product.md]]`,
> `#[[file:../../steering/tech.md]]`,
> `#[[file:../../steering/structure.md]]`,
> `#[[file:../../steering/patterns-and-reference-repos.md]]`.

## Definitions

- **Ready to generate** = CUDA available AND the Krea 2 Turbo diffusion
  checkpoint, the Qwen3-VL text encoder, and the Qwen-Image VAE are all present
  on disk and pass the presence check.
- **Job** = one generation request and its result(s): prompt, all parameters,
  seed, model id, timing, and output image path(s).
- **Tenant** = anything that wants GPU residency (Phase 1: the text encoder and
  the diffusion model).

---

## Requirement 1 — One-click install & environment

**User story:** As the owner, I want to install and launch the Studio with a
single script, so that setup is not a debugging session.

- WHEN the bootstrap script is run on a machine with Python 3.11 and Git present
  THE SYSTEM SHALL create a project-local virtual environment, install the
  exact-pinned dependencies, and launch the app without further manual steps.
- WHEN the app launches THE SYSTEM SHALL NOT perform any `git pull` or
  self-directed `pip install`/upgrade of its own environment.
- WHEN a dependency required to boot the shell is missing THE SYSTEM SHALL report
  the specific missing dependency in plain language and exit non-zero, rather
  than partially starting.
- WHEN the app starts THE SYSTEM SHALL detect the presence/absence of CUDA and
  record it for the readiness check, without failing if CUDA is absent (the UI
  still loads).

## Requirement 2 — Fast, model-free startup (lazy loading)

**User story:** As the owner, I want the UI to open immediately, so that I'm not
waiting on multi-gigabyte weights before I can do anything.

- WHEN the app starts THE SYSTEM SHALL construct and serve the full UI without
  loading any model weights into memory or onto the GPU.
- WHEN no generation has yet been requested THE SYSTEM SHALL keep GPU memory
  free of model weights.
- WHEN the first generation is requested AND the system is ready THE SYSTEM
  SHALL load the required model components at that point, and reuse them for
  subsequent generations without reloading.

## Requirement 3 — Model download (streaming, resumable, auto-placed)

**User story:** As the owner, I want to download the Krea 2 Turbo model from
inside the app with visible progress, so that I never have to hunt for files or
place them by hand.

- WHEN the owner triggers a model download THE SYSTEM SHALL stream
  human-readable progress updates to the UI as the download proceeds.
- WHEN a download is interrupted and re-triggered THE SYSTEM SHALL resume rather
  than restart from zero.
- WHEN a download completes THE SYSTEM SHALL place each file (diffusion
  checkpoint, text encoder, VAE) in the correct directory automatically.
- WHEN the Hugging Face endpoint is unreachable THE SYSTEM SHALL report that in
  plain language and SHALL NOT crash or hang indefinitely.
- WHEN required model files are already present THE SYSTEM SHALL detect this and
  report "already downloaded" rather than re-downloading.
- WHEN a file is present but incomplete or wrong-sized THE SYSTEM SHALL treat it
  as not-present for readiness purposes rather than assuming it is valid.

## Requirement 4 — Readiness surfacing (plain language, never a red wall)

**User story:** As the owner, I want the app to tell me exactly what's missing
before I generate, so that I'm never staring at a cryptic error.

- WHEN the system is not ready to generate THE SYSTEM SHALL display a
  plain-language banner stating the specific reason (e.g. "Krea 2 Turbo model
  not downloaded yet", "No CUDA GPU detected").
- WHEN the system becomes ready THE SYSTEM SHALL hide the not-ready banner.
- WHEN the owner attempts to generate while not ready THE SYSTEM SHALL refuse
  with the specific reason and SHALL NOT throw an unhandled error.
- WHEN any handler encounters an internal error THE SYSTEM SHALL surface a
  plain-language message to the UI and write the full traceback to a log file,
  never rendering a raw traceback in the UI.

## Requirement 5 — Image generation (Krea 2 Turbo)

**User story:** As the owner, I want to type a prompt, set a few parameters, and
get an image, so that the core creative loop works.

- WHEN the owner submits a prompt while the system is ready THE SYSTEM SHALL
  load the Qwen3-VL text encoder, encode the prompt using Krea's baked template,
  and then offload the text encoder before loading the diffusion model.
- WHEN encoding is complete THE SYSTEM SHALL load the Krea 2 Turbo diffusion
  model and sample using Turbo defaults (8 steps, CFG disabled, fixed mu/shift
  1.15) unless the owner has overridden the exposed parameters.
- WHEN sampling is complete THE SYSTEM SHALL decode the latents with the
  Qwen-Image VAE and display the resulting image in the UI.
- WHEN the owner sets a specific seed THE SYSTEM SHALL use it, and WHEN no seed
  is set THE SYSTEM SHALL generate and record one so the result is reproducible.
- WHEN the owner adjusts an exposed parameter (steps, seed, width, height, batch
  size, batch count, precision) THE SYSTEM SHALL apply it to the generation.
- THE SYSTEM SHALL expose sampler parameters as real UI controls with honest
  tooltips, and SHALL NOT expose Krea's internal prompt template as an editable
  field.

## Requirement 6 — Batch controls (Forge parity, honestly labeled)

**User story:** As the owner, I want batch size and batch count as separate
controls, so that I can queue work the way I do in Forge — without being misled
about speed.

- WHEN the owner sets a batch size > 1 THE SYSTEM SHALL generate that many images
  in a single pass, subject to VRAM.
- WHEN the owner sets a batch count > 1 THE SYSTEM SHALL run that many sequential
  batches.
- THE SYSTEM SHALL present batch size and batch count as distinct controls, each
  with a tooltip explaining the difference (size = simultaneous/VRAM-bound,
  count = sequential/queue-bound).
- WHEN a requested batch size would exceed available VRAM THE SYSTEM SHALL report
  this in plain language before generating, rather than OOM-crashing mid-run.

## Requirement 7 — VRAM tenant discipline

**User story:** As the owner, I want the app to manage GPU memory so it doesn't
OOM on my 24 GB card, so that generation is reliable.

- WHEN a model component is needed THE SYSTEM SHALL request GPU residency
  through the VRAM manager rather than moving tensors to the GPU directly from
  arbitrary modules.
- WHEN the text encoder has finished encoding THE SYSTEM SHALL release its GPU
  memory before the diffusion model is loaded.
- WHEN a new tenant requests residency while another heavyweight tenant is
  resident THE SYSTEM SHALL unload the resident one first (Phase 1 has at most
  one heavyweight tenant resident at a time).
- WHEN the owner selects a precision (bf16 or fp8_scaled) THE SYSTEM SHALL load
  the corresponding weights and reflect the change in VRAM footprint.

## Requirement 8 — Persistence & history

**User story:** As the owner, I want every generation saved with its settings so
I can look back and reuse them, so that nothing is lost when I close the tab.

- WHEN a generation completes THE SYSTEM SHALL persist the job (prompt, all
  parameters, seed, model id, timing) and the output image path(s) to SQLite.
- WHEN the owner opens the History tab THE SYSTEM SHALL list recent jobs with
  their prompt, thumbnail/reference, and key parameters.
- WHEN the owner selects a past job THE SYSTEM SHALL show its full parameters and
  allow loading them back into the Generate controls.
- WHEN the History tab lists jobs THE SYSTEM SHALL page results rather than
  loading the entire history into memory at once.
- WHEN the app is closed and reopened THE SYSTEM SHALL retain all prior jobs and
  images.

## Requirement 9 — Model-agnostic seam (present, single-entry)

**User story:** As a recovering architect, I want the model to sit behind a
registry from day one, so that adding a second model later doesn't require a
rewrite — without over-building now.

- THE SYSTEM SHALL route all model access (metadata, load, generate) through a
  model registry rather than referencing Krea 2 directly from the UI shell.
- WHEN the Krea 2 backend fails to import or load THE SYSTEM SHALL mark that
  backend unavailable with a plain-language reason and keep the rest of the app
  running.
- THE SYSTEM SHALL define the registry entry shape (checkpoints, VAE, text
  encoder, sampler defaults, precision options, VRAM tiers) such that a second
  backend can be added by providing a new entry and backend module, with no
  change to the UI shell.
- **Scope guard:** Phase 1 SHALL ship exactly one registry entry (Krea 2 Turbo).
  It SHALL NOT include stub or placeholder entries for other models.

---

## Non-functional requirements

- **Startup:** UI interactive in a few seconds on the target machine, with no
  model in memory (ties to R2).
- **Stability:** a failure in download, a model backend, or a single generation
  SHALL NOT crash the app process; it degrades to a reported state.
- **Reproducibility:** a completed job SHALL be reconstructable from the DB
  alone (ties to R8, R5).
- **Testability:** each pattern module (downloader, system_check, db,
  vram_manager, registry, krea2 backend) SHALL have pytest coverage of the
  behaviors above — specifically the "test obligations" in the patterns steering
  doc. The reference patterns were test-driven and must remain so.
- **Platform:** Windows primary, Linux secondary; no Windows-only path
  assumptions that break Linux.

## Explicit out-of-scope for Phase 1 (do not implement)

Prompt-optimizer LLM; LoRA training (any form); RAW checkpoint support;
ControlNet / img2img / inpainting; video; X/Y/Z plot; model-path federation;
upscalers; a second model backend; auto-update. Each has a home in a later
phase per `steering/product.md`.

## Open decisions to confirm with the owner (do not silently resolve)

1. Precision default for Phase 1: ship **fp8_scaled** as the default (more VRAM
   headroom, ~13 GB) or **bf16** (max quality, fits 24 GB)? Current assumption:
   offer both, default to fp8_scaled, clearly labeled.
2. Whether the History tab needs delete/export in Phase 1 or that waits. Current
   assumption: view + load-params only in Phase 1; no delete/export yet.
