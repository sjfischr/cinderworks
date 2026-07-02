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
- [ ] Negative prompts (Raw model; Turbo is CFG-free)
- [ ] Krea 2 Raw support (full-step, CFG) alongside Turbo
- [ ] Image-to-image (init image + denoise strength)
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
- [ ] LoRA loading (Krea 2 official + community LoRAs, weight slider)
- [ ] Upscaler model picker (4x-UltraSharp, RealESRGAN-anime, any
      spandrel-loadable file dropped into models_store)
- [ ] Hires-fix-style two-pass: generate → upscale → img2img refine
- [ ] Queue: fire-and-forget multiple jobs, cancel button
      (batch_count exists; needs a visible queue + cancellation)
- [ ] Gallery niceties: send-to-upscale, send-to-img2img, open output
      folder, compare A/B

### M4 — Robustness / distribution
- [ ] One-click install (bootstrap already exists; add torch-CUDA
      detection + version pinning sanity checks)
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
- **Inpainting / outpainting** with mask editor
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
