# Requirements Document

## Introduction

Cinderworks is a local-first, model-agnostic image generation studio with a Gradio web UI. Phase 1 covers the core creative loop: download Krea 2 Turbo, generate an image, persist the job/params/result, and recall history. The target user is a professional software developer who does image generation as a creative hobby. The product thesis is: "Works when you log in. Still works tomorrow."

This spec is scoped strictly to Phase 1. Prompt-optimizer LLM, LoRA training, additional model backends, ControlNet, img2img, inpainting, video, X/Y/Z plot, model-path federation, upscalers, and auto-update are all out of scope and belong in later phases.

## Glossary

- **Studio**: The Cinderworks image generation application (Gradio web UI).
- **System**: The Cinderworks application as a whole, including all modules.
- **Ready_to_Generate**: A state where CUDA is available AND the Krea 2 Turbo diffusion checkpoint, the Qwen3-VL text encoder, and the Qwen-Image VAE are all present on disk and pass the presence check.
- **Job**: One generation request and its results: prompt, all parameters, seed, model id, timing, and output image path(s).
- **Tenant**: Any component that wants GPU residency (Phase 1: the text encoder and the diffusion model).
- **VRAM_Manager**: The central coordinator that controls all GPU memory allocation and enforces tenant discipline.
- **Registry**: The model-agnostic routing layer through which all model access (metadata, load, generate) flows.
- **Bootstrap_Script**: The single entry-point script that creates the environment, installs dependencies, and launches the app.
- **Batch_Size**: The number of images generated in a single parallel pass (VRAM-bound).
- **Batch_Count**: The number of sequential batches to run (queue-bound).
- **Precision**: The floating-point format used for model weights (bf16 or fp8_scaled in Phase 1).
- **Turbo_Defaults**: 8 steps, CFG 0.0 (guidance disabled per Krea convention), fixed mu/shift 1.15.
- **Downloader**: The module responsible for streaming, resumable model downloads via huggingface_hub.
- **Readiness_Banner**: A plain-language UI element that communicates what is missing before generation can occur.

## Requirements

### Requirement 1: One-click Install and Environment

**User Story:** As the owner, I want to install and launch the Studio with a single script, so that setup is not a debugging session.

#### Acceptance Criteria

1. WHEN the Bootstrap_Script is run on a machine with Python 3.11+, Git, and uv present, THE System SHALL create a project-local virtual environment, install the exact-pinned dependencies from the committed lock file, and launch the Gradio server such that the UI is accessible at a localhost URL without further manual steps.
2. WHEN the app launches, THE System SHALL NOT perform any git pull or self-directed pip install or upgrade of its own environment.
3. IF a prerequisite required by the bootstrap script itself (Python 3.11+, Git, or uv) is not found, THEN THE System SHALL report which prerequisite is missing by name and exit non-zero without creating or modifying the virtual environment.
4. IF a Python dependency required to boot the shell is missing or fails to import after installation, THEN THE System SHALL report the specific missing dependency by package name in plain language and exit non-zero, rather than partially starting.
5. WHEN the app starts, THE System SHALL detect the presence or absence of CUDA, store the result in the readiness state used by the readiness banner, and continue startup regardless of the outcome so that the UI remains accessible when CUDA is absent.

### Requirement 2: Fast Model-Free Startup with Lazy Loading

**User Story:** As the owner, I want the UI to open immediately, so that I am not waiting on multi-gigabyte weights before I can do anything.

#### Acceptance Criteria

1. WHEN the app starts, THE System SHALL construct and serve the full UI without loading any model weights into memory or onto the GPU.
2. WHILE no generation has yet been requested, THE System SHALL keep GPU memory free of model weights and SHALL NOT import model-loading libraries that trigger CUDA initialization.
3. WHEN the first generation is requested AND the system is Ready_to_Generate, THE System SHALL load the required model components at that point, and reuse them for subsequent generations without reloading.
4. WHEN the app starts, THE System SHALL render the UI fully navigable (all tabs, controls, and event handlers responsive) within 5 seconds on the target machine.

### Requirement 3: Model Download (Streaming, Resumable, Auto-Placed)

**User Story:** As the owner, I want to download the Krea 2 Turbo model from inside the app with visible progress, so that I never have to hunt for files or place them by hand.

#### Acceptance Criteria

1. WHEN the owner triggers a model download, THE Downloader SHALL stream progress updates to the UI that include the current file name, percentage complete, and bytes downloaded versus total bytes for each file being downloaded.
2. WHEN a download is interrupted and re-triggered, THE Downloader SHALL resume from the last successfully received byte rather than restart from zero.
3. WHEN a download completes, THE Downloader SHALL place each file (diffusion checkpoint, text encoder, VAE) into the MODEL_DIR subfolder designated by the Config object, without requiring the owner to specify or confirm the destination.
4. IF the Hugging Face endpoint is unreachable or a download request receives no response within 30 seconds, THEN THE Downloader SHALL report the connectivity failure in plain language and SHALL NOT crash or block the UI.
5. WHEN required model files are already present and pass the size check, THE Downloader SHALL report "already downloaded" for each such file rather than re-downloading it.
6. WHEN a file is present on disk but its size does not match the expected size from the Hugging Face repository metadata, THE System SHALL treat it as not-present for readiness purposes rather than assuming it is valid.
7. IF a multi-file download partially fails (one or more files succeed but at least one fails), THEN THE Downloader SHALL report which specific file(s) failed, retain the successfully downloaded files, and allow the owner to re-trigger download for only the failed file(s).
8. WHILE a download is in progress, THE Downloader SHALL yield progress updates at least once per received chunk so the UI remains responsive and the owner can observe forward movement.

### Requirement 4: Readiness Surfacing (Plain Language, Never Cryptic)

**User Story:** As the owner, I want the app to tell me exactly what is missing before I generate, so that I am never staring at a cryptic error.

#### Acceptance Criteria

1. WHEN the system is not Ready_to_Generate, THE System SHALL display a Readiness_Banner listing all currently unmet conditions (e.g. "No CUDA GPU detected", "Krea 2 Turbo model not downloaded yet") so that each active reason is visible simultaneously.
2. WHEN the system becomes Ready_to_Generate, THE System SHALL hide the Readiness_Banner within 2 seconds of the condition being met.
3. WHEN a download completes or a model file becomes present, THE System SHALL re-evaluate readiness and update the Readiness_Banner accordingly without requiring an app restart.
4. WHEN the owner attempts to generate while not Ready_to_Generate, THE System SHALL display an inline error message prefixed with "❌" stating the specific unmet condition(s), SHALL NOT initiate any generation work, and SHALL NOT throw an unhandled error.
5. IF any handler encounters an internal error, THEN THE System SHALL display a plain-language message prefixed with "❌" in the UI describing the failure category (e.g. "not enough VRAM", "something went wrong"), SHALL write the full traceback to the log file, SHALL include the log file path in the displayed message, and SHALL NOT render a raw traceback or stack trace in the UI.

### Requirement 5: Image Generation (Krea 2 Turbo)

**User Story:** As the owner, I want to type a prompt, set a few parameters, and get an image, so that the core creative loop works.

#### Acceptance Criteria

1. WHEN the owner submits a prompt while the system is Ready_to_Generate, THE System SHALL load the Qwen3-VL text encoder, encode the prompt using the baked template with multi-layer hidden-state aggregation from the 12 selected layers, and then offload the text encoder before loading the diffusion model.
2. WHEN encoding is complete, THE System SHALL load the Krea 2 Turbo diffusion model and sample using Turbo_Defaults (8 steps, guidance_scale 0.0 disabled, fixed mu/shift 1.15) unless the owner has overridden the exposed parameters.
3. WHEN sampling is complete, THE System SHALL decode the latents with the Qwen-Image VAE and display the resulting image in the UI.
4. WHEN the owner sets a specific seed, THE System SHALL use that seed, and WHEN no seed is set, THE System SHALL generate a random seed, record it in the Job, and use it for generation so the result is reproducible.
5. WHEN the owner adjusts an exposed parameter (steps: 1–100, seed: 0–2^32-1, width: 512–2048 in multiples of 64, height: 512–2048 in multiples of 64, Batch_Size, Batch_Count, Precision), THE System SHALL validate the value is within bounds and apply it to the generation.
6. THE System SHALL expose sampler parameters as real UI controls with honest tooltips describing each parameter's effect, and SHALL NOT expose the internal prompt template as an editable field.
7. IF the owner submits an empty prompt, THEN THE System SHALL refuse the generation with a plain-language message indicating a prompt is required.
8. WHILE a generation is in progress, THE System SHALL yield progress updates (at minimum: encoding started, sampling step N of M, decoding) so the owner can observe forward movement and estimate remaining time.

### Requirement 6: Batch Controls (Honestly Labeled)

**User Story:** As the owner, I want Batch_Size and Batch_Count as separate controls, so that I can queue work without being misled about speed.

#### Acceptance Criteria

1. WHEN the owner sets a Batch_Size greater than 1, THE System SHALL generate that many images in a single parallel pass, where image *i* in the batch uses seed value (base_seed + i), and the total images produced equals Batch_Size × Batch_Count.
2. WHEN the owner sets a Batch_Count greater than 1, THE System SHALL run that many sequential batches, each producing Batch_Size images, reusing the same VRAM footprint as a single batch.
3. THE System SHALL present Batch_Size and Batch_Count as distinct controls, each defaulting to 1, with Batch_Size accepting integer values from 1 to 16 and Batch_Count accepting integer values from 1 to 100, each with a tooltip explaining the difference (size = simultaneous/VRAM-bound, count = sequential/queue-bound).
4. IF a requested Batch_Size would exceed available VRAM as estimated by the VRAM_Manager, THEN THE System SHALL report this in plain language before generating and refuse the request, rather than OOM-crashing mid-run.
5. IF an out-of-memory error occurs during batch generation despite passing the pre-generation VRAM check, THEN THE System SHALL halt generation, report the failure in plain language, and preserve any images already completed in prior batches.

### Requirement 7: VRAM Tenant Discipline

**User Story:** As the owner, I want the app to manage GPU memory so it does not OOM on my 24 GB card, so that generation is reliable.

#### Acceptance Criteria

1. WHEN a model component is needed, THE VRAM_Manager SHALL handle GPU residency through its acquire/release API rather than allowing arbitrary modules to move tensors to the GPU directly.
2. WHEN the text encoder has finished encoding, THE VRAM_Manager SHALL release its GPU memory (moving weights back to CPU) before the diffusion model is loaded, ensuring peak VRAM usage never includes both the encoder and DiT simultaneously.
3. WHEN a new Tenant requests residency while another heavyweight Tenant is resident, THE VRAM_Manager SHALL unload the resident Tenant first (Phase 1 enforces at most one heavyweight Tenant resident at a time).
4. WHEN the owner selects a Precision (bf16 or fp8_scaled), THE System SHALL load the corresponding weights variant and the VRAM_Manager SHALL update its available-memory estimate to reflect the new footprint.
5. IF the VRAM_Manager's acquire call fails due to insufficient memory after attempting to unload existing tenants, THEN THE System SHALL surface a plain-language OOM error to the UI suggesting the owner lower batch size or switch to fp8_scaled precision.

### Requirement 8: Persistence and History

**User Story:** As the owner, I want every generation saved with its settings so I can look back and reuse them, so that nothing is lost when I close the tab.

#### Acceptance Criteria

1. WHEN a generation completes, THE System SHALL persist the Job (prompt, all parameters, seed, model id, duration in milliseconds, status) and the output image path(s) to SQLite, storing parameters as a JSON blob in params_json and artifact metadata (path, seed, width, height) in the artifact table.
2. IF the database write fails during job persistence, THEN THE System SHALL surface a plain-language error to the UI indicating the save failed, retain the generated image in the output directory, and not discard the generation result.
3. WHEN the owner opens the History tab, THE System SHALL list Jobs ordered by creation time descending, displaying each Job's prompt (truncated to 120 characters), a thumbnail of the first output image, and the parameters: seed, steps, resolution, precision, and model id.
4. WHEN the owner selects a past Job, THE System SHALL display all stored parameters and provide a control that populates the Generate tab's prompt, seed, steps, width, height, precision, batch size, and batch count fields with that Job's values without triggering a generation.
5. WHEN the History tab lists Jobs, THE System SHALL page results in pages of 20 Jobs, loading the next page only when the owner scrolls to the end or activates a "load more" control.
6. WHEN the app is closed and reopened, THE System SHALL retain all prior Jobs and images, and IF a referenced image file is missing from disk, THEN THE System SHALL display a placeholder indicating the image is unavailable rather than failing to render the Job entry.

### Requirement 9: Model-Agnostic Seam (Present, Single-Entry)

**User Story:** As a recovering architect, I want the model to sit behind a Registry from day one, so that adding a second model later does not require a rewrite — without over-building now.

#### Acceptance Criteria

1. THE System SHALL route all model access (metadata, load, generate) through the Registry rather than referencing Krea 2 directly from the UI shell.
2. WHEN the app starts, THE System SHALL defer import of backend modules until a backend is first accessed, so that a failing backend module does not prevent app startup.
3. IF the Krea 2 backend fails to import or load, THEN THE System SHALL mark that backend unavailable with a plain-language reason and keep the rest of the app running.
4. IF a generation is requested against a backend marked unavailable, THEN THE System SHALL refuse the request with the recorded plain-language reason rather than attempting the import again.
5. THE System SHALL define the Registry entry shape (checkpoints, VAE, text encoder, sampler defaults, precision options, VRAM tiers) such that a second backend can be added by providing a new entry and backend module, with no change to the UI shell.
6. THE System SHALL ship exactly one Registry entry (Krea 2 Turbo) in Phase 1, with no stub or placeholder entries for other models.

### Requirement 10: Non-Functional — Startup Performance

**User Story:** As the owner, I want the app to feel instant when I open it, so that I can start thinking about images rather than waiting.

#### Acceptance Criteria

1. WHEN the app starts with no model loaded, THE System SHALL render the UI fully navigable (all tabs, controls, and event handlers responsive to user input) within 5 seconds on the target machine (Windows 11, i9, RTX 4090, NVMe storage).

### Requirement 11: Non-Functional — Stability and Graceful Degradation

**User Story:** As the owner, I want the app to stay running even when something goes wrong, so that a single failure does not force a restart.

#### Acceptance Criteria

1. IF a failure occurs in the Downloader, a model backend, or a single generation, THEN THE System SHALL continue running, display a plain-language explanation in the UI, log the full traceback to the log file, and remain responsive for new user actions.
2. IF a model backend raises an error on import, THEN THE System SHALL disable that backend with a plain-language reason visible in the UI and keep all other functionality (download, history, readiness checks) available.
3. IF a failure occurs during an in-progress generation, THEN THE System SHALL abort only that generation, preserve any previously persisted jobs, and return the Generate controls to an idle state ready for a new request.

### Requirement 12: Non-Functional — Reproducibility

**User Story:** As the owner, I want to reproduce any past generation from its saved record alone, so that I can iterate on results I liked.

#### Acceptance Criteria

1. THE System SHALL store sufficient information in each Job record (prompt, params_json, seed, model_id, precision) such that re-running the same inputs produces a bit-identical result given the same model weights, hardware, and PyTorch version.
2. WHEN the owner loads parameters from a past Job and regenerates on the same machine with the same model weights and PyTorch version, THE System SHALL produce an image bit-identical to the original.

### Requirement 13: Non-Functional — Testability

**User Story:** As the developer, I want every pattern module to have pytest coverage of its specified behaviors, so that regressions are caught before they reach the user.

#### Acceptance Criteria

1. THE System SHALL provide at least one pytest test case per acceptance criterion behavior for each pattern module: Downloader, system_check, db, VRAM_Manager, Registry, and krea2 backend.
2. THE System SHALL test behaviors defined in the acceptance criteria of Requirements 1–9 rather than implementation details such as internal data structures or private method signatures.

### Requirement 14: Non-Functional — Platform Compatibility

**User Story:** As the owner, I want the app to run on both Windows and Linux without path-related failures.

#### Acceptance Criteria

1. THE System SHALL use platform-agnostic path handling (pathlib or os.path) for all file operations, with no Windows-only path separators, drive-letter assumptions, or case-sensitivity assumptions that break Linux.
2. THE System SHALL pass its full test suite on both Windows and Linux without path-related failures.
