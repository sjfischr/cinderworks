---
inclusion: auto
---

# Product Context — Cinderworks

> This steering file governs all specs and implementation for the Cinderworks
> image studio. Treat it as authoritative product context.

## What Cinderworks Is

A local-first, model-agnostic image generation studio with a Gradio web UI.
It is a deliberate departure from the AUTOMATIC1111 / Forge lineage and the
opposite of ComfyUI's node-graph paradigm. The first model backend is Krea 2
(Turbo), but the architecture treats models as pluggable so future backends
(Flux, Qwen-Image, Z-Image, etc.) slot in behind a common interface.

## The Core Promise (Testable)

**"Works when you log in. Still works tomorrow."**

The target user is a professional software developer who does image generation
as a creative hobby. They spend all day in software and do not want their
hobby to be a second debugging job. They want a tool, not a toolkit. Every
product decision serves one of two properties:

1. **Works when you log in** — near-instant startup, one-click install, models
   download themselves with visible progress, plain-language readiness instead
   of cryptic errors, session/history persistence so nothing is lost.
2. **Still works tomorrow** — a pinned, locked environment; deliberate
   user-triggered updates (never auto-self-update); model backends decoupled so
   one broken integration can't brick the whole app.

## Who It Is NOT For

- People who want a node graph as the creative surface (that's ComfyUI).
- People who need the deep A1111 extension ecosystem.
- Commercial/hosted multi-tenant use (this is a local single-user tool).

## Product Name

Do not hard-code a product name in source. Read it from a single `APP_NAME`
constant in `config.py`. The working title is "Cinderworks."

## Testable Success Criterion (Phase 1 Gate)

Phase 1 succeeds if, after two weeks of real use, the owner reaches for the
Studio instead of ComfyUI for everyday Krea 2 generation without thinking
about it.

## Phasing (Strict — Do Not Build Ahead)

Phase boundaries are load-bearing. Nothing ships until the previous phase is
solid and used.

- **Phase 1 — Core loop (THIS SPEC):** download Krea 2 Turbo → generate an
  image → persist the job/params/result → recall history. Model-registry
  abstraction present but holds exactly one entry. One-click install. Glass UI
  shell.
- **Phase 2 — Second model backend:** add one more model to prove the registry
  abstraction is real. Own spec.
- **Phase 3 — Prompt optimizer LLM:** lazy-loaded, user-selectable local LLM
  that rewrites prompts, opt-in per generation, unloads before the image model
  runs. Own spec.
- **Phase 4 — In-app LoRA training:** subprocess-wrapped trainer. Own spec.
- **Deferred:** ControlNet, video, X/Y/Z plot, model-path federation, embedded
  trainer.

## Explicit Out-of-Scope for Phase 1

Prompt-optimizer LLM; LoRA training (any form); RAW checkpoint support;
ControlNet / img2img / inpainting; video; X/Y/Z plot; model-path federation;
upscalers; a second model backend; auto-update.
