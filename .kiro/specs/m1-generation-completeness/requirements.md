# Requirements Document

## Introduction

M1 Generation Completeness extends Cinderworks Studio beyond the Phase 1 core creative loop to deliver the remaining generation-time features needed for Forge Neo parity. This milestone adds multi-LoRA stacking, model checkpoint selection, image-to-image with inpainting, one-click upscaling from the gallery, and keyboard navigation for the image gallery. Together these capabilities close the gap between Cinderworks and the weekly-use features a Forge Neo user depends on.

This spec assumes Phase 1 is complete: text-to-image generation, job persistence, history with load-params, model download, readiness surfacing, and VRAM tenant discipline are all working. M1 builds on top of those modules without rewriting them.

## Glossary

- **Studio**: The Cinderworks image generation application (Gradio web UI).
- **System**: The Cinderworks application as a whole, including all modules.
- **LoRA**: A Low-Rank Adaptation weight file (.safetensors) that modifies the diffusion model's behavior at generation time without replacing the base checkpoint.
- **LoRA_Stack**: The ordered list of LoRA entries (file path + weight) selected for a generation request.
- **LoRA_Weight**: A decimal multiplier (0.0–2.0) controlling the strength of a single LoRA's influence on generation.
- **LoRA_Panel**: The collapsible UI section in the Generate tab where the user manages LoRA selection and weights.
- **Checkpoint**: A complete model weight file (diffusion DiT) representing a specific model variant and precision.
- **Checkpoint_Selector**: The dropdown control in the Generate tab that selects the active model checkpoint for generation.
- **Active_Checkpoint**: The checkpoint currently selected in the Checkpoint_Selector, used on the next generation request.
- **Img2Img**: Image-to-image generation where an existing image is used as the initialization for the diffusion process, controlled by a denoise strength parameter.
- **Denoise_Strength**: A float value (0.0–1.0) controlling how much of the init image is preserved versus regenerated. 0.0 preserves the image entirely; 1.0 ignores it completely.
- **Mask**: A binary image (same dimensions as the init image) where painted regions indicate areas to regenerate and unpainted regions indicate areas to preserve.
- **Mask_Editor**: The brush-based drawing tool overlaid on the init image that allows the user to paint a mask.
- **Inpainting**: A variant of Img2Img where only the masked region is regenerated and the unmasked region is preserved from the init image.
- **Upscaler**: The spandrel-based Real-ESRGAN pipeline that enlarges images (existing from Phase 1).
- **Gallery**: The image display area showing generation results, supporting selection and navigation.
- **Focused_Image**: The currently selected/highlighted image in the Gallery that receives keyboard navigation actions.
- **Registry**: The model-agnostic routing layer through which all model access flows (from Phase 1).
- **RegistryEntry**: The data structure defining a model's metadata, backend module, checkpoints, sampler defaults, and VRAM tiers.
- **VRAM_Manager**: The central coordinator that controls all GPU memory allocation and enforces tenant discipline (from Phase 1).
- **Krea2Pipeline**: The diffusers pipeline class used for Krea 2 inference.
- **Raw_Model**: The Krea 2 Raw checkpoint variant using full sampling (28 steps, CFG 4.5).
- **Turbo_Model**: The Krea 2 Turbo checkpoint variant using accelerated sampling (8 steps, CFG 0.0).
- **Turbo_Defaults**: 8 steps, CFG 0.0 (guidance disabled), fixed mu/shift 1.15.
- **Raw_Defaults**: 28 steps, CFG 4.5, fixed mu/shift 1.15.

## Requirements

### Requirement 1: LoRA Auto-Detection from Folder

**User Story:** As the owner, I want the app to automatically find all LoRA files in my loras folder, so that I can drop files in and use them without configuration.

#### Acceptance Criteria

1. WHEN the LoRA_Panel is opened or refreshed, THE System SHALL scan the configured loras directory and list every file with a `.safetensors` extension as an available LoRA.
2. THE System SHALL resolve the loras directory path from the Config object, defaulting to `studio/loras/` relative to the project root.
3. IF the configured loras directory does not exist, THEN THE System SHALL create it on first access and display an empty LoRA list with a guidance message indicating where to place LoRA files. IF directory creation fails due to permissions or disk space, THEN THE System SHALL still display the guidance message.
4. IF a `.safetensors` file in the loras directory is not a valid LoRA (fails to parse headers), THEN THE System SHALL skip that file with a warning logged rather than crashing the scan or hiding all other valid LoRAs.
5. WHEN a new `.safetensors` file is added to the loras directory, THE System SHALL detect it on the next panel refresh without requiring an app restart.

### Requirement 2: Multi-LoRA Stacking with Weighted Application

**User Story:** As the owner, I want to load multiple LoRAs simultaneously with individual weight controls, so that I can blend styles and subjects from different training runs.

#### Acceptance Criteria

1. THE LoRA_Panel SHALL allow the user to add one or more LoRAs to the LoRA_Stack via a dropdown populated from the auto-detected list and an "Add LoRA" button.
2. THE LoRA_Panel SHALL display each entry in the LoRA_Stack with the LoRA filename, a weight slider (0.0–2.0, default 1.0, step 0.05), and a remove button.
3. WHEN the user adds the same LoRA file that is already in the LoRA_Stack, THE System SHALL refuse the addition and display a message indicating the LoRA is already loaded.
4. WHEN a generation is requested with a non-empty LoRA_Stack, THE System SHALL apply each LoRA in stack order to the pipeline with its configured weight before sampling begins.
5. WHEN a generation is requested with an empty LoRA_Stack, THE System SHALL run generation without any LoRA modifications applied.
6. THE System SHALL load LoRAs fresh for each generation and unload them after sampling completes, rather than persistently fusing LoRA weights into the base model.
7. IF a LoRA file in the LoRA_Stack fails to load at generation time (corrupted file, incompatible architecture), THEN THE System SHALL abort the generation with a plain-language error identifying the specific LoRA that failed, rather than producing a corrupted result.
8. WHEN the LoRA_Stack is non-empty, THE System SHALL record the full LoRA_Stack (file paths and weights) in the Job's params_json so that the combination is reproducible from history.

### Requirement 3: Model Checkpoint Selector

**User Story:** As the owner, I want to choose which model checkpoint to use from a dropdown in the Generate tab, so that I can switch between Turbo and Raw variants without editing config files.

#### Acceptance Criteria

1. THE Checkpoint_Selector SHALL appear in the Generate tab as a dropdown listing all available checkpoints with display labels formatted as "{model display name} {precision}" (e.g. "Krea 2 Turbo fp8", "Krea 2 Raw bf16").
2. WHEN the app starts, THE Checkpoint_Selector SHALL populate its options from the Registry by querying all registered model entries and their precision options, requiring no hardcoded UI list.
3. WHEN a new RegistryEntry is added to the Registry with a corresponding backend module, THE Checkpoint_Selector SHALL include that model's checkpoints in its dropdown without any UI code changes.
4. WHEN the user selects a different checkpoint in the Checkpoint_Selector, THE System SHALL NOT reload the pipeline immediately — the new checkpoint SHALL be applied on the next generation request.
5. WHEN a generation is requested with the Raw_Model selected, THE System SHALL apply Raw_Defaults (28 steps, CFG 4.5) as the base sampler parameters unless the user has explicitly overridden them.
6. WHEN a generation is requested with the Turbo_Model selected, THE System SHALL apply Turbo_Defaults (8 steps, CFG 0.0) as the base sampler parameters unless the user has explicitly overridden them.
7. WHEN the user switches from one checkpoint to another between generations, THE System SHALL unload the previous checkpoint's pipeline and load the new one via the VRAM_Manager on the next generation request.
8. THE System SHALL store the selected checkpoint identifier (model_id + precision) in the Job record so that history entries reflect which checkpoint produced each result.

### Requirement 4: Image-to-Image Generation

**User Story:** As the owner, I want to push a generated image into an img2img workflow, so that I can refine results iteratively using the same diffusion model.

#### Acceptance Criteria

1. WHEN the user activates "Send to img2img" on a Gallery image, THE System SHALL navigate to the img2img section and display the selected image as the init image.
2. THE img2img section SHALL provide a Denoise_Strength slider (0.0–1.0, default 0.5, step 0.05) controlling how much of the init image is preserved versus regenerated.
3. THE img2img section SHALL provide the same prompt, seed, steps, width, height, and precision controls as the txt2img Generate tab.
4. WHEN the user submits an img2img generation request, THE System SHALL encode the init image into latent space, add noise at the level determined by Denoise_Strength, and sample from that noised latent using the active checkpoint's sampler parameters.
5. WHEN Denoise_Strength is 0.0, THE System SHALL return the init image unchanged without running the sampling loop.
6. WHEN Denoise_Strength is 1.0, THE System SHALL perform full sampling from noise (equivalent to txt2img) while still using the init image dimensions.
7. THE System SHALL persist img2img jobs to the database with the same schema as txt2img jobs, additionally storing the init image path and Denoise_Strength in params_json.
8. IF no init image is set when the user attempts img2img generation, THEN THE System SHALL refuse the generation with a plain-language message indicating an init image is required.

### Requirement 5: Inpainting with Mask Editor

**User Story:** As the owner, I want to paint a mask on my image to selectively regenerate parts of it, so that I can fix specific regions without affecting the rest.

#### Acceptance Criteria

1. THE img2img section SHALL include a Mask_Editor that allows the user to paint directly on the displayed init image using a brush tool.
2. THE Mask_Editor SHALL provide brush size control (adjustable radius) so the user can paint fine details or broad areas.
3. WHEN the user paints on the init image, THE Mask_Editor SHALL render the painted region as a visible semi-transparent overlay so the user can see what is masked.
4. THE Mask_Editor SHALL provide a clear button that removes all painted mask content, resetting to an unmasked state.
5. WHEN a generation is submitted with a painted mask, THE System SHALL regenerate only the masked region while preserving the unmasked region pixel-for-pixel from the init image.
6. WHEN a generation is submitted without any mask painted, THE System SHALL treat the entire image as the region to denoise (standard img2img behavior per Requirement 4).
7. THE System SHALL store the mask data (or a reference to the saved mask image) in the Job's params_json so that the inpainting operation is recorded in history.
8. IF the mask dimensions do not match the init image dimensions, THEN THE System SHALL resize the mask to match the init image before applying, rather than failing with a dimension mismatch error.

### Requirement 6: Upscale from Gallery

**User Story:** As the owner, I want to upscale any generated image with one click from the gallery, so that I can enlarge results without leaving the generation workflow.

#### Acceptance Criteria

1. WHEN the user activates "Send to Upscale" on a Gallery image, THE System SHALL submit the selected image to the existing spandrel-based Real-ESRGAN upscaler pipeline.
2. WHEN upscaling completes, THE System SHALL display the upscaled image in the Gallery alongside or replacing the original, with a visual indicator that it is an upscaled result.
3. WHEN upscaling completes, THE System SHALL save the upscaled image to the outputs directory and persist it as an artifact in the job record linked to the original generation job. IF saving the image file fails, THEN THE System SHALL still persist the artifact record.
4. WHILE an upscale operation is in progress, THE System SHALL display a progress indicator so the user knows the operation is running.
5. IF the upscaler model is not available (not downloaded or failed to load), THEN THE System SHALL report the unavailability in plain language rather than crashing or producing a blank result. THE System SHALL still allow visual display of any previously generated results.
6. THE System SHALL coordinate upscaler GPU memory through the VRAM_Manager, acquiring the upscaler tenant before processing and releasing it after completion.

### Requirement 7: Gallery Keyboard Navigation

**User Story:** As the owner, I want to navigate through generated images using arrow keys, so that I can browse results quickly without reaching for the mouse.

#### Acceptance Criteria

1. WHEN the Gallery has focus and the user presses the right arrow key, THE System SHALL advance the Focused_Image to the next image in the gallery sequence.
2. WHEN the Gallery has focus and the user presses the left arrow key, THE System SHALL move the Focused_Image to the previous image in the gallery sequence.
3. WHEN the Focused_Image is on the last image and the user presses the right arrow key, THE System SHALL remain on the last image (no wrap-around).
4. WHEN the Focused_Image is on the first image and the user presses the left arrow key, THE System SHALL remain on the first image (no wrap-around).
5. THE Gallery SHALL display a visible indicator (border, highlight, or outline) on the Focused_Image so the user can identify which image is currently selected.
6. WHEN the Gallery receives focus (via tab or click), THE System SHALL set the Focused_Image to the first image if no image was previously focused.
7. THE Gallery keyboard navigation SHALL work with both single-image results and multi-image batch results.

### Requirement 8: LoRA and Checkpoint Interaction with VRAM Management

**User Story:** As the owner, I want LoRA loading and checkpoint switching to respect VRAM discipline, so that switching models or adding LoRAs does not crash the app.

#### Acceptance Criteria

1. WHEN a generation with LoRAs is requested, THE System SHALL load LoRA weights after the diffusion model is acquired on GPU and before sampling begins, within the same VRAM_Manager tenant session.
2. WHEN a generation with LoRAs completes, THE System SHALL unload the LoRA weights from the pipeline before releasing the acquired diffusion model, leaving the base model unmodified for subsequent generations. THE System SHALL retain the VRAM_Manager tenant session.
3. WHEN the user switches checkpoints between generations, THE VRAM_Manager SHALL release the previously loaded checkpoint's GPU memory before loading the new checkpoint, ensuring peak VRAM never includes both checkpoints simultaneously.
4. IF loading a LoRA would cause the combined model to exceed available VRAM as estimated by the VRAM_Manager, THEN THE System SHALL refuse the generation with a plain-language message suggesting fewer LoRAs or lower precision.
5. WHEN LoRAs are applied to either the Raw_Model or the Turbo_Model, THE System SHALL accept the same LoRA files for both variants (Krea 2 convention: train on Raw, apply on Turbo or Raw).

### Requirement 9: Send-to Workflow Integration

**User Story:** As the owner, I want "Send to" actions in the gallery that push images to img2img or upscale with one click, so that my workflow flows like Forge Neo without manual file management.

#### Acceptance Criteria

1. THE Gallery SHALL display "Send to img2img" and "Send to Upscale" action controls on each generated image when the image is selected or hovered.
2. WHEN "Send to img2img" is activated, THE System SHALL populate the img2img section's init image with the selected gallery image and navigate the user to the img2img section.
3. WHEN "Send to Upscale" is activated, THE System SHALL immediately submit the selected image for upscaling without requiring additional user confirmation.
4. THE send-to actions SHALL be accessible via both mouse interaction (button click) and keyboard shortcut while the image is focused in the Gallery.
5. IF the selected image file is missing from disk (deleted externally) WHEN the user activates a send-to operation, THEN THE System SHALL display a plain-language error indicating the image is unavailable rather than crashing the send-to operation.

### Requirement 10: Non-Functional — LoRA Loading Performance

**User Story:** As the owner, I want LoRA application to be fast enough that it does not noticeably delay generation beyond the base inference time.

#### Acceptance Criteria

1. WHEN a single LoRA is applied, THE System SHALL complete LoRA weight fusion in under 3 seconds on the target machine (RTX 4090, NVMe storage) before sampling begins.
2. WHEN multiple LoRAs are stacked (up to 5 simultaneously), THE System SHALL complete all LoRA weight fusions in under 10 seconds total on the target machine before sampling begins.
3. THE System SHALL NOT reload the base checkpoint from disk when applying or removing LoRAs between generations — only the LoRA delta weights are loaded fresh per generation.

### Requirement 11: Non-Functional — Checkpoint Switch Latency

**User Story:** As the owner, I want switching checkpoints to be reasonably fast, so that experimenting with Raw vs Turbo does not feel like restarting the app.

#### Acceptance Criteria

1. WHEN switching between checkpoints of the same precision (e.g. Turbo fp8 to Raw fp8), THE System SHALL complete the pipeline reload in under 15 seconds on the target machine.
2. WHEN switching between checkpoints of different precisions (e.g. Turbo fp8 to Turbo bf16), THE System SHALL complete the pipeline reload in under 30 seconds on the target machine.
3. WHILE a checkpoint reload is in progress, THE System SHALL display a status message indicating the model is loading, so the user understands the delay.

### Requirement 12: Non-Functional — Img2Img and Inpainting Consistency

**User Story:** As the owner, I want img2img and inpainting results to be reproducible from their saved parameters.

#### Acceptance Criteria

1. THE System SHALL store sufficient information in each img2img Job record (prompt, params_json including denoise_strength, seed, model_id, precision, init_image_path, mask reference) such that re-running the same inputs produces a bit-identical result given the same model weights, hardware, and PyTorch version. This applies to all img2img operations including those with zero denoise strength.
2. WHEN the user loads parameters from a past img2img Job and regenerates on the same machine with the same model weights, init image, mask, and PyTorch version, THE System SHALL produce an image bit-identical to the original.

