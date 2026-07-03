# Cinderworks Roadmap

Goal: a Gradio-based local image generation studio that is a genuine
alternative to Forge Neo — simpler to install, honest about failures,
and fast on consumer hardware — starting from the Krea 2 backend that
already works today.

## Where we are (July 2026)

Working today:

- Krea 2 Turbo text-to-image, fp8_scaled, fully GPU-resident on 24 GB
  cards (~1.4 s/step at 1024×1024)
- VRAM tenant discipline with honest pre-flight refusals (no silent
  WDDM shared-memory thrash — the failure mode that plagues every
  Windows diffusion stack)
- Model downloader with resume, per-file status, multi-repo sources
- Job history (SQLite) with paging, multi-delete (DB + files),
  load-params-from-job
- Upscaling: Lanczos built-in, Real-ESRGAN 4x via spandrel (tiled,
  VRAM-coordinated)
- Live per-step progress with stall watchdog + diagnostics

## MVP: "formidable Forge Neo alternative"

The bar: a Forge Neo user with a 12–24 GB NVIDIA card can switch and
not miss anything they use weekly. Roughly ordered:

### M1 — Generation completeness

- [ ] **LoRA loading (multi-stack, weighted)** — load one or more LoRAs
      from a `loras/` folder, each with a decimal weight value (0.0–2.0).
      Follow the exact Forge Neo pattern: stack multiple LoRAs, apply at
      generation time, unload when switching. Krea 2 official LoRAs +
      community LoRAs supported. UI: list of loaded LoRAs with weight
      sliders, add/remove buttons.
- [ ] **Model checkpoint selector** — dropdown in the Generate tab to
      choose the active model. Phase 1: all Krea 2 family checkpoints
      (Turbo fp8, Turbo bf16, Raw fp8, Raw bf16). Wiring must support
      non-Krea models (e.g. zImage) via the existing registry — adding
      a new model entry + backend module should populate the dropdown
      automatically with no UI code changes.
- [ ] **Image-to-image with inpainting** — push any generated image
      directly to an img2img/inpaint workflow from the gallery. Includes:
      init image input, denoise strength slider, mask editor (brush-based
      inpainting). Same "send to" pattern as Forge Neo.
- [ ] **Upscale from gallery** — push a generated image directly to the
      upscaler from the generation screen (one-click enlarge). Uses the
      existing spandrel-based Real-ESRGAN pipeline.
- [ ] **Gallery keyboard navigation** — scroll through generated images
      using left/right arrow keys. Focus the gallery and use keyboard
      to browse without mouse.
- [ ] Negative prompts (Raw model; Turbo is CFG-free)
- [ ] Krea 2 Raw support (full-step, CFG) alongside Turbo
- [ ] Seed variation tools: reuse seed from gallery image, increment,
      random-batch
- [ ] PNG metadata embed (params in the file, Forge/A1111-compatible
      format) + drag-a-PNG-to-restore-params

### M2 — Memory tiers for smaller cards (the Forge "GPU Weights" story)

- [ ] Streamed group offloading fallback (diffusers
      `enable_group_offload(leaf_level, use_stream=True)`) so 12–16 GB
      cards run fp8 and 24 GB cards can run bf16 — slower but working,
      never thrashing
- [ ] GGUF quantized checkpoint support (Q4–Q8) for 8 GB cards
- [ ] Per-tier auto-selection with the existing pre-flight math (the
      VRAM plan already computes encode/sampling peaks — route to the
      right tier instead of refusing)

### M3 — Quality-of-life parity

- [ ] Upscaler model picker (4x-UltraSharp, RealESRGAN-anime, any
      spandrel-loadable file dropped into models_store)
- [ ] Hires-fix-style two-pass: generate → upscale → img2img refine
- [ ] Queue: fire-and-forget multiple jobs, cancel button
      (batch_count exists; needs a visible queue + cancellation)
- [ ] Gallery niceties: compare A/B, open output folder

### M4 — Robustness / distribution

- [ ] One-click install improvements (torch-CUDA detection + version
      pinning sanity checks beyond what bootstrap already does)
- [ ] Settings tab: output dir, VRAM reserve override, precision
      default, attention backend picker
- [ ] Crash-free session restore (queue + params survive restart)

## Beyond MVP (spitballing, unordered)

- **More model families** — the registry was built for this: Z-Image
  Turbo (small + fast, great for 8 GB), Qwen-Image (text rendering),
  FLUX family. Each is one RegistryEntry + one backend module.
- **SageAttention / FlashAttention backends** — attention backend
  picker; sage is the current speed king on 40-series
- **torch.compile** on the transformer blocks (composes with fp8;
  musubi-tuner reports meaningful step-time wins)
- **Prompt assistant** — local LLM (the Qwen3-VL encoder is already a
  VLM!) for prompt expansion/rewrites; Phase 3/4 tenant slots exist in
  the VRAM manager for exactly this
- **Outpainting** with mask editor (extend canvas beyond borders)
- **ControlNet-style conditioning** as Krea 2 ecosystem support lands
- **LoRA training** (musubi-tuner already supports Krea 2; wrap the
  train-on-Raw / infer-on-Turbo loop)
- **API mode** — headless REST endpoint (Gradio ships one for free;
  document + stabilize it)
- **Multi-GPU** — device_map placement for the encoder on a second card
- **Style presets** — the official Krea LoRA pack with trigger words
  as one-click styles
- **Telemetry-driven auto-tuning** — we already log real peaks; use
  them to recalibrate activation estimates per resolution on each
  machine (self-profiling instead of constants)

## Principles (unchanged from the design doc)

1. Honest failures over silent degradation — refuse with a plain
   message rather than thrash.
2. One GPU chokepoint — everything is a tenant of the shared
   VRAMManager.
3. The shell stays thin — backends do the work; a new model must never
   require UI surgery.
4. "Still works tomorrow" — pinned deps, no surprise upgrades.

## LoRA Implementation Reference (Forge Neo Pattern)

The LoRA implementation follows the Forge Neo approach:

- **Storage:** `studio/loras/` folder (configurable via Config). Any
  `.safetensors` LoRA file dropped in is auto-detected.
- **UI:** A collapsible panel in the Generate tab with:
  - File browser / dropdown to select LoRAs from the folder
  - "Add LoRA" button (supports multiple stacked LoRAs)
  - Per-LoRA weight slider (0.0 to 2.0, default 1.0)
  - Remove button per LoRA
- **Application:** LoRAs are fused into the pipeline at generation time
  via `pipe.load_lora_weights()` / `pipe.fuse_lora()`. Multiple LoRAs
  stack additively (Forge Neo behavior).
- **Lifecycle:** LoRAs are loaded fresh per generation (not persistently
  fused). Changing LoRA selection between generations does not require
  pipeline reload.
- **Krea 2 convention:** Train on Raw, apply on Turbo. Both model
  variants accept the same LoRA files.
