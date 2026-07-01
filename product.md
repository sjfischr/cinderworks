# Product Overview

> **Working title:** TBD. "AvatarForge" is rejected (dated). Referred to as
> **"the Studio"** throughout these docs until a name is chosen. Do not
> hard-code a product name in source; read it from a single `APP_NAME` constant.

## What the Studio is

A local-first, model-agnostic image generation studio with a Gradio web UI.
It is a deliberate departure from the AUTOMATIC1111 / Forge lineage and the
opposite of ComfyUI's node-graph paradigm. The first model backend is Krea 2
(Turbo), but the architecture treats models as pluggable so future backends
(Flux, Qwen-Image, Z-Image, etc.) slot in behind a common interface.

## The core promise (this is the product thesis, and it is testable)

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

## Who it is NOT for

- People who want a node graph as the creative surface (that's ComfyUI, and
  it's good at that — we are not competing on flexibility-through-nodes).
- People who need the deep A1111 extension ecosystem.
- Commercial/hosted multi-tenant use (this is a local single-user tool).

## Why this is worth building (honest version)

The justification is **personal utility**, not market domination. ComfyUI has
100k+ GitHub stars; the Studio will not out-popularity it and should not try.
What the Studio does is package a set of already-proven patterns (from the
owner's BeatBunny and Higgs Studio projects) that solve real, well-documented
pain the incumbents have left unsolved for years — most notably session
persistence, which has been the top-requested missing A1111 feature since 2022.
The owner already has working code for these patterns. This is repackaging a
known-good solution into a new domain, not researching an unknown one.

## Testable success criterion (the gate)

Phase 1 succeeds if, after two weeks of real use, the owner reaches for the
Studio instead of ComfyUI for everyday Krea 2 generation without thinking
about it. If they open ComfyUI out of habit anyway, Phase 1 has failed and
later phases should not be built on top of it. Every later phase gets its own
small testable claim before it earns its complexity.

## Phasing (strict — do not build ahead)

Phase boundaries are load-bearing. Getting a basic UI that can download the
model and perform a generation is the first rung, and nothing else ships until
that rung is solid and used.

- **Phase 1 — Core loop (THIS SPEC):** download Krea 2 Turbo → generate an
  image → persist the job/params/result → recall history. Model-registry
  abstraction present but holds exactly one entry. One-click install. Glass UI
  shell. This is `specs/image-studio-core`.
- **Phase 2 — Second model backend:** add one more model (e.g. Flux or
  Qwen-Image) to prove the registry abstraction is real. Own spec, own
  Layer-3-style source-verified research first.
- **Phase 3 — Prompt optimizer LLM:** lazy-loaded, user-selectable local LLM
  that rewrites prompts, opt-in per generation, unloads before the image model
  runs. Own spec.
- **Phase 4 — In-app LoRA training:** subprocess-wrapped trainer (NOT an
  embedded training loop for v1). Own spec.
- **Deferred / revisit-only:** ControlNet, video, X/Y/Z plot, model-path
  federation, embedded (non-subprocess) trainer. See
  `#[[file:../../spec-research/layer4-gap-analysis.md]]`.

## Detailed research backing every decision here

The four research layers are the evidentiary basis for this product and must be
treated as authoritative context:

- `#[[file:../../spec-research/layer1-forge-neo-and-competitive.md]]`
- `#[[file:../../spec-research/layer2-reference-patterns.md]]`
- `#[[file:../../spec-research/layer3-krea2-inference-interface.md]]`
- `#[[file:../../spec-research/layer4-gap-analysis.md]]`
