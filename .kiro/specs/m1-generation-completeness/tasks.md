# Implementation Plan: M1 Generation Completeness

## Overview

This plan implements the five generation-time features for Cinderworks Studio M1: multi-LoRA stacking, model checkpoint selection, image-to-image with inpainting, one-click upscaling from gallery, and keyboard navigation for gallery. All tasks build on the Phase 1 architecture (registry routing, VRAM tenant discipline, lazy model loading, SQLite persistence) and follow the existing layering (UI → Core → Models → DB). Python with Hypothesis PBT is used throughout.

## Tasks

- [x] 1. Database schema extensions and core data layer
  - [x] 1.1 Extend db/db.py with artifact_type and source_artifact_id columns
    - Add migration to add `artifact_type TEXT DEFAULT 'generated'` column to artifact table
    - Add migration to add `source_artifact_id INTEGER REFERENCES artifact(id)` column to artifact table
    - Update artifact creation functions to accept `artifact_type` and `source_artifact_id` params
    - Ensure existing artifacts default to `artifact_type='generated'`
    - _Requirements: 6.3, 15_
  - [x]* 1.2 Write property test for upscaled artifact linking (Property 15)
    - **Property 15: Upscaled artifact links to source**
    - For any upscale operation on a gallery image belonging to job J, the resulting artifact record SHALL have `artifact_type='upscaled'` and `source_artifact_id` pointing to the original artifact from job J.
    - **Validates: Requirements 6.3**
  - [x]* 1.3 Write property test for generation params round-trip (Property 6)
    - **Property 6: Generation params round-trip (persistence completeness)**
    - For any generation job (txt2img or img2img), serializing params_json and deserializing it SHALL produce a dict containing all required fields for reproduction.
    - **Validates: Requirements 2.8, 3.8, 4.7, 5.7, 12.1**

- [x] 2. LoRA discovery and management module
  - [x] 2.1 Create core/lora_manager.py with LoRAEntry, LoRAStack dataclasses and scan_loras()
    - Implement `LoRAEntry` dataclass (file_path, filename, weight)
    - Implement `LoRAStack` dataclass (entries list)
    - Implement `scan_loras(loras_dir: Path) -> list[str]` that scans for .safetensors files
    - Handle missing directory creation (with permissions fallback per Requirement 1.3)
    - Implement `validate_lora_file(file_path: Path) -> bool` for header validation
    - Skip invalid files with warning logged (Requirement 1.4)
    - Resolve loras directory path from Config object, defaulting to `studio/loras/`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x]* 2.2 Write property test for LoRA scan (Property 1)
    - **Property 1: LoRA scan returns exactly the valid safetensors files**
    - For any directory containing a mix of files with various extensions and validity states, `scan_loras()` SHALL return exactly those files that have a `.safetensors` extension AND pass header validation.
    - **Validates: Requirements 1.1, 1.4**
  - [x] 2.3 Implement LoRA stack operations (add, remove, duplicate rejection)
    - Add function to add LoRA to stack with duplicate check (same file path)
    - Add function to remove LoRA from stack by filename
    - Return updated stack JSON after each operation
    - Refuse duplicate addition with informational message
    - _Requirements: 2.1, 2.2, 2.3_
  - [x]* 2.4 Write property test for duplicate LoRA rejection (Property 2)
    - **Property 2: Duplicate LoRA rejection preserves stack**
    - For any LoRA_Stack and any LoRA file already present, attempting to add the same file SHALL be refused and the stack SHALL remain unchanged.
    - **Validates: Requirements 2.3**
  - [x] 2.5 Implement apply_loras() and unload_loras() with VRAM coordination
    - Implement `apply_loras(pipeline, stack)` using diffusers `load_lora_weights` in stack order
    - Implement `unload_loras(pipeline)` to restore base model weights
    - Integrate VRAM budget check (`can_fit()`) before application, summing base model + LoRA estimates
    - Raise RuntimeError with plain-language message identifying the failed LoRA on error
    - Load fresh per generation, unload after sampling completes
    - _Requirements: 2.4, 2.6, 2.7, 8.1, 8.2, 8.4, 8.5, 10.3_
  - [x]* 2.6 Write property test for LoRA stack order preservation (Property 3)
    - **Property 3: LoRA stack order and weights preserved through application**
    - For any non-empty LoRA_Stack with N entries, the pipeline's LoRA application function SHALL receive exactly N LoRAs in the same order with the same weights.
    - **Validates: Requirements 2.4, 8.1**
  - [x]* 2.7 Write property test for LoRA lifecycle (Property 4)
    - **Property 4: LoRA lifecycle — load fresh, unload after**
    - For any generation with a non-empty LoRA_Stack, after generation completes, the pipeline SHALL have zero LoRA modifications applied.
    - **Validates: Requirements 2.6, 8.2**
  - [x]* 2.8 Write property test for failed LoRA identification (Property 5)
    - **Property 5: Failed LoRA identified in error message**
    - For any LoRA_Stack where exactly one entry references a corrupt file, the error message SHALL contain the filename of that specific LoRA.
    - **Validates: Requirements 2.7**
  - [x]* 2.9 Write property test for VRAM overflow pre-flight (Property 13)
    - **Property 13: VRAM overflow pre-flight refuses generation**
    - For any generation request where estimated combined footprint exceeds budget, generation SHALL be refused with a plain-language message before any GPU loading begins.
    - **Validates: Requirements 8.4**
  - [x]* 2.10 Write property test for LoRA cross-model compatibility (Property 14)
    - **Property 14: LoRA cross-model compatibility**
    - For any valid LoRA file trained on Krea 2, `apply_loras()` SHALL succeed on both krea2-turbo and krea2-raw pipelines.
    - **Validates: Requirements 8.5**
  - [x]* 2.11 Write property test for no base checkpoint reload on LoRA ops (Property 16)
    - **Property 16: Base checkpoint not reloaded on LoRA operations**
    - For any sequence of LoRA apply and unload operations without checkpoint change, the base checkpoint pipeline SHALL remain cached in memory.
    - **Validates: Requirements 10.3**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Registry extension and checkpoint selection logic
  - [x] 4.1 Add Krea 2 Raw RegistryEntry to models/registry.py
    - Add `krea2-raw` RegistryEntry with display_name "Krea 2 Raw", 28 steps, CFG 4.5, mu_shift 1.15
    - Add both bf16 and fp8_scaled precision options with VRAM tier estimates
    - _Requirements: 3.1, 3.2, 3.3_
  - [x] 4.2 Implement list_checkpoint_options() in registry.py
    - Return all model+precision combinations as list of dicts with model_id, precision, display_label
    - Format display_label as "{display_name} {precision}" (e.g. "Krea 2 Turbo fp8_scaled")
    - Dynamically query all registered entries (no hardcoded list)
    - _Requirements: 3.1, 3.2, 3.3_
  - [x]* 4.3 Write property test for checkpoint label formatting (Property 9)
    - **Property 9: Checkpoint label formatting**
    - For any RegistryEntry with display_name D and precision option P, the label SHALL be formatted as "{D} {P}".
    - **Validates: Requirements 3.1**
  - [x] 4.4 Implement lazy checkpoint switching in krea2 backend
    - Extend `_get_pipeline` to accept `model_id` parameter and cache per `(model_id, precision, mode)`
    - Evict prior cached pipeline on checkpoint change
    - Coordinate through VRAM_Manager: release old checkpoint before loading new
    - Apply sampler defaults from RegistryEntry (Raw: 28/4.5, Turbo: 8/0.0) unless user overrides
    - _Requirements: 3.4, 3.5, 3.6, 3.7, 8.3_
  - [x]* 4.5 Write property test for sampler defaults follow model_id (Property 7)
    - **Property 7: Sampler defaults follow model_id**
    - For any generation request without explicit step/cfg overrides, resolved params SHALL match the RegistryEntry defaults for the selected model_id.
    - **Validates: Requirements 3.5, 3.6**
  - [x]* 4.6 Write property test for checkpoint switch releases before acquiring (Property 8)
    - **Property 8: Checkpoint switch releases before acquiring**
    - For any two consecutive generations with different checkpoints, the VRAM_Manager's release of the first SHALL complete before acquire of the second.
    - **Validates: Requirements 3.7, 8.3**

- [x] 5. Image-to-image and inpainting pipeline
  - [x] 5.1 Extend krea2.py generate() for img2img mode
    - Accept `init_image_path` in params; when present, trigger img2img mode
    - Encode init image to latent space using the model's VAE
    - Add noise at the level determined by `denoise_strength` (0.0–1.0)
    - When denoise_strength is 0.0, return init image unchanged without sampling
    - When denoise_strength is 1.0, perform full sampling from noise using init image dimensions
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [x] 5.2 Implement mask composite for inpainting
    - Accept `mask_path` in params; when present, apply inpainting logic
    - Resize mask to match init image dimensions if they differ (Requirement 5.8)
    - After generation, composite: output[unmasked] = init_image[unmasked] pixel-for-pixel
    - When no mask is provided, treat entire image as the region to denoise (standard img2img)
    - _Requirements: 5.5, 5.6, 5.8_
  - [x]* 5.3 Write property test for inpainting preserves unmasked pixels (Property 10)
    - **Property 10: Inpainting preserves unmasked pixels**
    - For any init image and any binary mask, every pixel whose mask value is unmasked SHALL be bit-identical to the corresponding pixel in the init image.
    - **Validates: Requirements 5.5**
  - [x]* 5.4 Write property test for mask resize (Property 11)
    - **Property 11: Mask resize to match init image dimensions**
    - For any mask with dimensions different from the init image, the resized mask SHALL have the same dimensions as the init image.
    - **Validates: Requirements 5.8**
  - [x] 5.5 Persist img2img/inpainting job data
    - Store init_image_path, denoise_strength, mask_path in params_json
    - Store model_id + precision in params_json
    - Ensure all fields needed for reproducibility are persisted (Requirement 12.1)
    - _Requirements: 4.7, 5.7, 12.1, 12.2_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. UI layer — LoRA panel and checkpoint selector
  - [x] 7.1 Implement LoRA panel in ui/controls.py
    - Create `create_lora_panel()` returning dropdown (available LoRAs), weight slider (0.0–2.0, default 1.0, step 0.05), add button, and stack display (Dataframe)
    - Display each stack entry with filename, weight, and remove button
    - Wire refresh button to `on_refresh_loras()` handler
    - _Requirements: 2.1, 2.2_
  - [x] 7.2 Implement checkpoint selector in ui/controls.py
    - Create `create_checkpoint_selector()` returning a Dropdown populated from `registry.list_checkpoint_options()`
    - Populate on app start from Registry query (no hardcoded values)
    - Selection change does NOT trigger immediate reload (lazy-switch)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [x] 7.3 Implement img2img controls in ui/controls.py
    - Create `create_img2img_controls()` returning init image display, denoise slider (0.0–1.0, default 0.5, step 0.05), and mask editor (brush tool with adjustable size, clear button, semi-transparent overlay)
    - Provide same prompt, seed, steps, width, height, precision controls as txt2img
    - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4_
  - [x]* 7.4 Write unit tests for UI controls
    - Test LoRA panel component creation and initial state
    - Test checkpoint selector population from registry
    - Test img2img controls with default values
    - _Requirements: 2.1, 2.2, 3.1, 4.2, 5.1_

- [x] 8. UI layer — Gallery with keyboard navigation and send-to actions
  - [x] 8.1 Create ui/gallery.py with gallery component and send-to buttons
    - Implement `create_gallery_with_actions()` returning Gallery with "Send to img2img" and "Send to Upscale" action buttons
    - Display action controls on selected/hovered images
    - Support single-image and multi-image batch results
    - _Requirements: 9.1, 9.2, 9.3, 7.7_
  - [x] 8.2 Implement keyboard navigation JS for gallery
    - Inject JavaScript for ArrowLeft/ArrowRight navigation
    - Clamp behavior: no wrap-around at boundaries (first stays first, last stays last)
    - Visual focus indicator (2px solid accent border) on focused thumbnail
    - Set focus to first image when gallery receives focus with no prior selection
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_
  - [x]* 8.3 Write property test for gallery keyboard navigation clamping (Property 12)
    - **Property 12: Gallery keyboard navigation is clamped**
    - For any gallery with N images and focused index I, right arrow → min(I+1, N-1), left arrow → max(I-1, 0).
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.7**

- [x] 9. UI handlers — generation, send-to, and LoRA operations
  - [x] 9.1 Implement on_generate_img2img handler in ui/handlers.py
    - Accept prompt, steps, seed, width, height, precision, init_image, denoise_strength, mask_data, lora_stack_json, checkpoint_id
    - Validate init image is present (refuse with message if missing)
    - Pass params to registry.run_generation() with img2img mode
    - Follow Phase 1 error boundary pattern (catch, log, user-friendly message)
    - _Requirements: 4.1, 4.4, 4.8, 5.5, 5.6_
  - [x] 9.2 Implement send-to handlers in ui/handlers.py
    - `on_send_to_img2img(gallery_selection)` — extract image path, return for init image component, navigate to img2img section
    - `on_send_to_upscale(gallery_selection)` — submit selected image to upscaler pipeline immediately
    - Handle missing file case with plain-language error (Requirement 9.5)
    - Coordinate upscaler GPU memory through VRAM_Manager (acquire tenant, release after)
    - Display progress indicator during upscale operation
    - Save upscaled image to outputs directory, persist artifact with type='upscaled' and source_artifact_id
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 9.3 Implement LoRA management handlers in ui/handlers.py
    - `on_refresh_loras()` — rescan loras directory, return updated dropdown choices
    - `on_add_lora(lora_name, current_stack_json)` — add to stack, reject duplicates
    - `on_remove_lora(lora_name, current_stack_json)` — remove from stack
    - Wire LoRA stack into generation params (record in params_json for reproducibility)
    - _Requirements: 1.1, 1.5, 2.1, 2.3, 2.8_
  - [x] 9.4 Wire keyboard shortcuts for send-to actions
    - Make send-to actions accessible via keyboard shortcut while image is focused in Gallery
    - _Requirements: 9.4_
  - [x]* 9.5 Write unit tests for handlers
    - Test on_generate_img2img with missing init image (error message)
    - Test on_send_to_img2img returns correct image path
    - Test on_send_to_upscale with missing file (error message)
    - Test on_add_lora duplicate rejection
    - Test on_refresh_loras returns updated list
    - _Requirements: 4.8, 9.2, 9.5, 2.3, 1.5_

- [x] 10. Integration wiring — app.py assembly
  - [x] 10.1 Wire all new UI components into app.py Gradio Blocks
    - Add LoRA panel to Generate tab
    - Add checkpoint selector to Generate tab
    - Add img2img section (tab or accordion) with controls and mask editor
    - Replace or extend existing gallery with `create_gallery_with_actions()`
    - Wire all event handlers (generate, send-to, LoRA ops, refresh)
    - Connect checkpoint selector change to store selected checkpoint_id (no reload)
    - Store model_id + precision in job record on each generation
    - _Requirements: 2.1, 3.1, 3.4, 3.8, 4.1, 9.1_
  - [x]* 10.2 Write integration tests for end-to-end workflows
    - Test img2img workflow: send-to → populate → generate → persist → display
    - Test inpainting workflow: paint mask → generate → composite → persist
    - Test upscale from gallery: select → submit → upscale → persist artifact → display
    - Test checkpoint switch: select Raw → generate (28 steps) → select Turbo → generate (8 steps)
    - Test LoRA + checkpoint combination: stack LoRAs + select model → generate → verify params stored
    - _Requirements: 4.1, 5.5, 6.1, 3.5, 3.6, 2.4, 2.8_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design (16 total)
- Unit tests validate specific examples and edge cases
- The design uses Python with Hypothesis for PBT — all code in this plan is Python
- Performance tests (Requirements 10, 11) are hardware-specific and not included as tasks; they should be run manually on the RTX 4090 target machine

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.2", "2.3", "4.1"] },
    { "id": 2, "tasks": ["2.4", "2.5", "4.2"] },
    { "id": 3, "tasks": ["2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "4.3", "4.4"] },
    { "id": 4, "tasks": ["4.5", "4.6", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.5", "7.1", "7.2"] },
    { "id": 6, "tasks": ["5.3", "5.4", "7.3", "7.4", "8.1"] },
    { "id": 7, "tasks": ["8.2", "8.3", "9.1"] },
    { "id": 8, "tasks": ["9.2", "9.3", "9.4"] },
    { "id": 9, "tasks": ["9.5", "10.1"] },
    { "id": 10, "tasks": ["10.2"] }
  ]
}
```
