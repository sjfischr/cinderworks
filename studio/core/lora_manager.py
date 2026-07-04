"""LoRA discovery, validation, stack management, and pipeline application.

This module handles scanning the loras directory for .safetensors files,
validating their headers, providing dataclasses for LoRA stack management,
and applying/unloading LoRA weights to the diffusion pipeline at generation time.

LoRAs are loaded fresh before each generation and unloaded after — they
are never persistently fused into the base model.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studio.config import Config

log = logging.getLogger(__name__)

# Estimated per-LoRA VRAM overhead in bytes. Real LoRA files for Krea 2
# are typically 100–500 MB. We use 300 MB as a conservative average for
# VRAM budget pre-flight checks.
_LORA_VRAM_ESTIMATE_BYTES = 300_000_000  # ~300 MB per LoRA


@dataclass
class LoRAEntry:
    """A single LoRA in the stack."""

    file_path: Path  # Absolute path to .safetensors file
    filename: str  # Display name (stem of the file)
    weight: float = 1.0  # 0.0–2.0, default 1.0


@dataclass
class LoRAStack:
    """Ordered list of LoRA entries for a generation."""

    entries: list[LoRAEntry] = field(default_factory=list)


def get_loras_dir() -> Path:
    """Resolve the loras directory path from Config, defaulting to studio/loras/.

    Returns:
        Absolute path to the loras directory.
    """
    return Config.LORAS_DIR


def validate_lora_file(file_path: Path) -> bool:
    """Check that a .safetensors file has a parseable header.

    The safetensors format starts with an 8-byte little-endian uint64
    indicating the header length, followed by that many bytes of JSON
    metadata. We validate that we can read and parse that header.

    Args:
        file_path: Path to the .safetensors file to validate.

    Returns:
        True if the header is parseable, False on corruption or I/O error.
    """
    try:
        with open(file_path, "rb") as f:
            # Read the 8-byte header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                log.warning(
                    "LoRA file too small to contain header: %s",
                    file_path.name,
                )
                return False

            header_len = struct.unpack("<Q", header_len_bytes)[0]

            # Sanity check: header shouldn't be larger than the file
            # or unreasonably large (> 100 MB is suspicious for a header)
            if header_len == 0:
                log.warning(
                    "LoRA file has zero-length header: %s",
                    file_path.name,
                )
                return False

            if header_len > 100_000_000:
                log.warning(
                    "LoRA file header length unreasonably large (%d bytes): %s",
                    header_len,
                    file_path.name,
                )
                return False

            # Read and parse the JSON header
            header_bytes = f.read(header_len)
            if len(header_bytes) < header_len:
                log.warning(
                    "LoRA file truncated (header claims %d bytes, got %d): %s",
                    header_len,
                    len(header_bytes),
                    file_path.name,
                )
                return False

            # Attempt JSON parse — this is the core validity check
            json.loads(header_bytes)
            return True

    except (OSError, IOError) as e:
        log.warning("Could not read LoRA file %s: %s", file_path.name, e)
        return False
    except (json.JSONDecodeError, struct.error, ValueError) as e:
        log.warning("Invalid LoRA header in %s: %s", file_path.name, e)
        return False


def scan_loras(loras_dir: Path) -> list[str]:
    """Scan directory for valid .safetensors LoRA files.

    Creates the directory if it does not exist (with permissions fallback).
    Skips invalid files with a warning logged rather than crashing.

    Args:
        loras_dir: Path to the directory to scan.

    Returns:
        Sorted list of display names (file stems) for valid LoRA files.
    """
    # Create directory if missing (Requirement 1.3)
    if not loras_dir.exists():
        try:
            loras_dir.mkdir(parents=True, exist_ok=True)
            log.info("Created loras directory: %s", loras_dir)
        except (PermissionError, OSError) as e:
            log.warning(
                "Could not create loras directory %s: %s. "
                "Returning empty LoRA list.",
                loras_dir,
                e,
            )
            return []

    # Scan for .safetensors files
    valid_names: list[str] = []

    try:
        candidates = sorted(loras_dir.glob("*.safetensors"))
    except (PermissionError, OSError) as e:
        log.warning("Could not list files in loras directory %s: %s", loras_dir, e)
        return []

    for file_path in candidates:
        if not file_path.is_file():
            continue

        if validate_lora_file(file_path):
            valid_names.append(file_path.stem)
        else:
            log.warning(
                "Skipping invalid LoRA file: %s (failed header validation)",
                file_path.name,
            )

    log.info("Found %d valid LoRA(s) in %s", len(valid_names), loras_dir)
    return valid_names


# ---------------------------------------------------------------------------
# Pipeline application — apply/unload LoRA weights for generation
# ---------------------------------------------------------------------------


def estimate_lora_stack_vram(stack: LoRAStack) -> int:
    """Estimate the total VRAM footprint of a LoRA stack.

    Uses a conservative per-LoRA estimate since exact sizes vary
    depending on the rank and architecture of each LoRA.

    Args:
        stack: The LoRA stack to estimate.

    Returns:
        Estimated VRAM in bytes for all LoRAs in the stack.
    """
    return len(stack.entries) * _LORA_VRAM_ESTIMATE_BYTES


# Adapter names become torch ModuleDict keys inside PEFT. ModuleDict
# rejects names containing "." (and PEFT is picky beyond that), so a file
# stem like "detail-tweaker-v1.5" would crash load_lora_weights on the
# spot. Whitelist [0-9A-Za-z_]; the lora_{i}_ prefix keeps names unique
# even when sanitization collides two stems.
_ADAPTER_NAME_UNSAFE = re.compile(r"[^0-9A-Za-z_]")

# Forge Neo aborts a LoRA when more than half its keys don't match the
# model ("[LORA] LoRA mismatch"); below that it warns and loads the rest.
_UNMATCHED_ABORT_RATIO = 0.5

# Recognized per-tensor suffixes across LoRA dialects. kohya/musubi use
# lora_down/lora_up (+ alpha); diffusers-PEFT uses lora_A/lora_B.
_LORA_KEY_SUFFIXES: dict[str, str] = {
    ".lora_A.weight": "A",
    ".lora_B.weight": "B",
    ".lora_down.weight": "A",
    ".lora_up.weight": "B",
    ".alpha": "alpha",
}


def _sanitize_adapter_name(raw: str) -> str:
    """Make a string safe to use as a PEFT adapter name."""
    return _ADAPTER_NAME_UNSAFE.sub("_", raw)


def _kohya_key_map(transformer: Any) -> dict[str, str]:
    """Map flattened kohya-style module names to real module paths.

    kohya/musubi checkpoints flatten module paths by replacing "." with
    "_" (e.g. "blocks.0.attn.to_q" -> "blocks_0_attn_to_q"), which cannot
    be reversed by string rules alone ("to_q" keeps its underscore).
    Forge and ComfyUI solve this by enumerating the actual model's module
    names and matching against the flattened form — same approach here.

    Returns an empty dict if the transformer can't be enumerated (tests
    with mock pipelines, or no transformer attached).
    """
    try:
        return {
            name.replace(".", "_"): name
            for name, _ in transformer.named_modules()
            if name
        }
    except Exception:
        return {}


def _split_lora_key(key: str) -> tuple[str, str] | None:
    """Split a state-dict key into (module path, part) where part is
    'A', 'B', or 'alpha'. Returns None for unrecognized suffixes."""
    for suffix, part in _LORA_KEY_SUFFIXES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    return None


# Module renames between the original Krea 2 checkpoint layout (llama-style
# names, what musubi-tuner and ComfyUI-ecosystem LoRAs are trained against)
# and the diffusers port (Krea2Transformer2DModel):
#
#   original                        diffusers
#   blocks.N                    ->  transformer_blocks.N
#   blocks.N.attn.wq            ->  transformer_blocks.N.attn.to_q
#   blocks.N.attn.wk            ->  transformer_blocks.N.attn.to_k
#   blocks.N.attn.wv            ->  transformer_blocks.N.attn.to_v
#   blocks.N.attn.wo            ->  transformer_blocks.N.attn.to_out.0
#   blocks.N.attn.gate          ->  transformer_blocks.N.attn.to_gate
#   blocks.N.mlp.{gate,up,down} ->  transformer_blocks.N.ff.{gate,up,down}
#
# The weights are identical 1:1 (QKV is not fused in the diffusers port),
# so this is a pure rename — same trick diffusers' own non-diffusers
# checkpoint converters apply for Flux/SDXL LoRAs.
_ORIGINAL_ATTN_LEAF_RENAMES: dict[str, str] = {
    "attn.wq": "attn.to_q",
    "attn.wk": "attn.to_k",
    "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0",
    "attn.gate": "attn.to_gate",
}


def _translate_original_layout(path: str) -> str:
    """Translate an original-Krea-2-layout module path to diffusers names.

    No-op for paths already in diffusers form: "transformer_blocks..."
    does not start with the "blocks" segment, diffusers uses ".ff." not
    ".mlp.", and the attention leaf names differ in every case.
    """
    segments = path.split(".")
    if segments and segments[0] == "blocks":
        segments[0] = "transformer_blocks"
    out = ".".join(segments)
    out = out.replace(".mlp.", ".ff.")
    for old, new in _ORIGINAL_ATTN_LEAF_RENAMES.items():
        if out.endswith("." + old) or out == old:
            out = out[: -len(old)] + new
            break
    return out


def _translate_original_layout_flat(flat: str) -> str:
    """Same rename for kohya-flattened names ("blocks_0_attn_wq")."""
    out = flat
    if out.startswith("blocks_"):
        out = "transformer_blocks_" + out[len("blocks_") :]
    out = out.replace("_mlp_", "_ff_")
    for old, new in _ORIGINAL_ATTN_LEAF_RENAMES.items():
        old_flat = old.replace(".", "_")
        if out.endswith("_" + old_flat):
            out = out[: -len(old_flat)] + new.replace(".", "_")
            break
    return out


def _normalize_module_path(
    base: str,
    kohya_map: dict[str, str],
    real_modules: set[str] | None = None,
) -> str | None:
    """Normalize a LoRA key's module path to diffusers 'transformer.*' form.

    Handles the dialects seen in the wild:
    - diffusers-PEFT:   transformer.transformer_blocks.0.attn.to_q
    - ComfyUI/musubi:   diffusion_model.blocks.0.attn.wq (original layout)
    - kohya flattened:  lora_unet_blocks_0_attn_wq

    Original-layout module names (wq/wk/wv/wo, attn.gate, mlp.*) are
    translated to the diffusers port's names. When `real_modules` (the
    live transformer's module paths) is provided, the final path is
    validated against it — an unvalidated path handed to PEFT crashes
    the whole load with "Target modules ... not found in the base
    model", taking every other key down with it.

    Returns None when the path can't be resolved against the model.
    """
    for flat_prefix in ("lora_unet_", "lora_transformer_"):
        if base.startswith(flat_prefix):
            flat = base[len(flat_prefix) :]
            # kohya_map values come from the live model's named_modules,
            # so they are real by construction — no translation/validation.
            real = kohya_map.get(flat) or kohya_map.get(
                _translate_original_layout_flat(flat)
            )
            return f"transformer.{real}" if real else None

    if base.startswith("transformer."):
        rest = base[len("transformer.") :]
    elif base.startswith("diffusion_model."):
        rest = base[len("diffusion_model.") :]
    elif "." in base:
        # Bare module path with no component prefix
        rest = base
    else:
        return None

    rest = _translate_original_layout(rest)
    if real_modules is not None and rest not in real_modules:
        return None
    return "transformer." + rest


def _convert_lora_state_dict(
    state_dict: dict[str, Any],
    kohya_map: dict[str, str],
    real_modules: set[str] | None = None,
) -> tuple[dict[str, Any], list[str], int]:
    """Convert a LoRA state dict to diffusers-PEFT format (Forge Neo style).

    Renames lora_down/lora_up to lora_A/lora_B, remaps ComfyUI and
    kohya-flattened module paths to 'transformer.*', translates
    original-checkpoint module names (wq/wo/mlp.*) to the diffusers
    port's names, and folds each kohya 'alpha' into its lora_B tensor
    (delta_W = (alpha/rank)·up·down). When `real_modules` is provided,
    every emitted module path is validated against the live model.

    Returns:
        (converted_dict, unmatched_keys, text_encoder_keys_skipped).
        Unmatched keys are dropped, not fatal — the caller decides
        whether the mismatch ratio warrants aborting.
    """
    tensors: dict[tuple[str, str], Any] = {}
    alphas: dict[str, float] = {}
    unmatched: list[str] = []
    te_skipped = 0

    for key, value in state_dict.items():
        split = _split_lora_key(key)
        if split is None:
            unmatched.append(key)
            continue
        base, part = split
        if base.startswith(("lora_te", "text_encoder.")):
            # This app applies LoRAs to the transformer only; text-encoder
            # weights (kohya lora_te_*) are skipped, matching Forge's
            # UNet/CLIP split where the CLIP half simply isn't loaded.
            te_skipped += 1
            continue
        norm = _normalize_module_path(base, kohya_map, real_modules)
        if norm is None:
            unmatched.append(key)
            continue
        if part == "alpha":
            try:
                alphas[norm] = float(value)
            except (TypeError, ValueError):
                log.warning("Ignoring non-scalar alpha for '%s'", norm)
        else:
            tensors[(norm, part)] = value

    converted: dict[str, Any] = {}
    for (norm, part), tensor in tensors.items():
        if part == "B" and norm in alphas:
            # lora_up is (out_features, rank); scale by alpha/rank so the
            # folded dict needs no separate alpha entries.
            rank = tensor.shape[1] if getattr(tensor, "ndim", 2) >= 2 else tensor.shape[0]
            if rank:
                tensor = tensor * (alphas[norm] / rank)
        converted[f"{norm}.lora_{part}.weight"] = tensor

    return converted, unmatched, te_skipped


def _prepare_lora_source(file_path: Path, pipeline: Any) -> str | dict[str, Any]:
    """Pre-read and normalize a LoRA file for diffusers (best effort).

    Follows the Forge Neo pattern: read the state dict ourselves, detect
    the key dialect, remap to what the model expects, warn on small
    mismatches, and abort only when most keys don't match. If the file
    can't be pre-read (safetensors unavailable, or exotic formats), the
    raw path is returned and diffusers' own loader gets to try.

    Raises:
        ValueError: If more than half the keys don't match the model —
            the LoRA was almost certainly trained for a different base.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        return str(file_path)

    try:
        state_dict = load_file(str(file_path), device="cpu")
    except Exception as e:
        log.warning(
            "Could not pre-read LoRA '%s' (%s) — passing path to diffusers directly",
            file_path.name,
            e,
        )
        return str(file_path)

    kohya_map = _kohya_key_map(getattr(pipeline, "transformer", None))
    # The live model's module paths, for validating every emitted key.
    # PEFT hard-fails the entire load if even one target module doesn't
    # exist, so anything unvalidated must be dropped, not passed through.
    real_modules = set(kohya_map.values()) if kohya_map else None
    converted, unmatched, te_skipped = _convert_lora_state_dict(
        state_dict, kohya_map, real_modules
    )

    if te_skipped:
        log.info(
            "LoRA '%s': skipped %d text-encoder key(s) (transformer-only application)",
            file_path.name,
            te_skipped,
        )

    considered = len(converted) + len(unmatched)
    if considered == 0 or len(unmatched) > considered * _UNMATCHED_ABORT_RATIO:
        raise ValueError(
            f"LoRA key format mismatch: {len(unmatched)} of {considered} keys "
            f"do not match the Krea 2 transformer (sample: "
            f"{unmatched[:3] if unmatched else 'no LoRA keys found'}). "
            f"This LoRA was likely trained for a different base model."
        )
    if unmatched:
        log.warning(
            "LoRA '%s': %d of %d keys unmatched — loading the remaining keys "
            "(sample unmatched: %s)",
            file_path.name,
            len(unmatched),
            considered,
            unmatched[:3],
        )

    return converted


def _cast_lora_layers_to_compute_dtype(pipeline: Any) -> None:
    """Keep injected LoRA weights in bf16 when the base model is fp8.

    Forge Neo forces "fp16 LoRA" whenever the base model is fp8/GGUF
    quantized — LoRA math in a storage-only float8 dtype either crashes
    (no fp8 matmul kernels) or destroys the adapter's precision. Our
    fp8_scaled mode applies layerwise casting hooks to the transformer;
    if PEFT initializes adapter weights from an fp8-stored base weight,
    this puts them back in the bf16 compute dtype. Best-effort no-op
    everywhere else (bf16 mode, mock pipelines, no torch).
    """
    try:
        import torch

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            return
        fixed = 0
        for name, param in transformer.named_parameters():
            if "lora_" in name and param.dtype == torch.float8_e4m3fn:
                param.data = param.data.to(torch.bfloat16)
                fixed += 1
        if fixed:
            log.info(
                "Cast %d LoRA weight tensor(s) from fp8 storage to bf16 "
                "compute dtype (fp8 base model)",
                fixed,
            )
    except Exception:
        log.debug("LoRA dtype normalization skipped", exc_info=True)


def _lora_failure_message(filename: str, exc: Exception) -> str:
    """Build a plain-language error naming the LoRA and the real cause.

    The original implementation reported every failure as "corrupted or
    incompatible", which hid the actual (and common) cause — the peft
    package missing entirely — behind a misleading message.
    """
    text = str(exc)
    if "peft" in text.lower():
        return (
            f"Could not load LoRA '{filename}' — the 'peft' package is not "
            f"installed. LoRA support requires it: run the bootstrap script "
            f"again (or `uv pip install peft`) and restart the app."
        )
    return (
        f"Could not load LoRA '{filename}' — {type(exc).__name__}: {text} "
        f"(if this LoRA was trained for a different base model, it cannot "
        f"be applied to Krea 2; remove it from the stack and try again)."
    )


def apply_loras(pipeline: Any, stack: LoRAStack, vram_manager: Any = None) -> None:
    """Apply LoRA stack to pipeline in order using diffusers load_lora_weights.

    Each LoRA is loaded fresh and applied with its configured weight.
    The function coordinates with the VRAM manager to pre-check that the
    combined footprint (base model + LoRAs) fits within the budget.

    Loading emulates Forge Neo: the state dict is pre-read and its key
    dialect (diffusers-PEFT, ComfyUI, kohya/musubi) normalized against
    the actual model, small key mismatches are warned about rather than
    fatal, failures name the real cause, and adapter weights are kept in
    the bf16 compute dtype when the base model is fp8-quantized.

    This function is called AFTER the diffusion model tenant is acquired
    on GPU and BEFORE sampling begins (Requirement 8.1).

    Args:
        pipeline: A diffusers pipeline object with load_lora_weights support.
        stack: The ordered LoRA stack to apply.
        vram_manager: Optional VRAMManager instance for VRAM budget checks.
            If None, the shared app-wide manager is used.

    Raises:
        RuntimeError: If VRAM budget is exceeded or any LoRA fails to load.
            The error message identifies the specific LoRA that failed
            and the underlying reason.
    """
    if not stack.entries:
        log.debug("Empty LoRA stack — nothing to apply")
        return

    # VRAM pre-flight check (Requirement 8.4)
    if vram_manager is None:
        from studio.core.vram_manager import get_vram_manager

        vram_manager = get_vram_manager()

    lora_vram = estimate_lora_stack_vram(stack)
    # The base model is already resident when this is called, so we check
    # whether the additional LoRA overhead fits in the remaining budget.
    # can_fit() checks against total capacity (after potential eviction),
    # but since we need the base model AND the LoRAs simultaneously, we
    # check that both fit together.
    resident = vram_manager.resident
    base_bytes = resident.estimated_bytes if resident else 0
    combined = base_bytes + lora_vram

    if not vram_manager.can_fit(combined):
        n_loras = len(stack.entries)
        raise RuntimeError(
            f"Not enough VRAM for {n_loras} LoRA{'s' if n_loras != 1 else ''} — "
            f"try removing LoRAs or switching to fp8_scaled."
        )

    log.info(
        "Applying %d LoRA(s) to pipeline (estimated additional VRAM: %d MB)",
        len(stack.entries),
        lora_vram // (1024 * 1024),
    )

    # Apply each LoRA in stack order (Requirement 2.4)
    adapter_names: list[str] = []
    adapter_weights: list[float] = []

    for i, entry in enumerate(stack.entries):
        adapter_name = _sanitize_adapter_name(f"lora_{i}_{entry.filename}")
        log.info(
            "Loading LoRA %d/%d: '%s' (weight=%.2f, adapter='%s')",
            i + 1,
            len(stack.entries),
            entry.filename,
            entry.weight,
            adapter_name,
        )

        try:
            source = _prepare_lora_source(entry.file_path, pipeline)
            pipeline.load_lora_weights(
                source,
                adapter_name=adapter_name,
            )
            adapter_names.append(adapter_name)
            adapter_weights.append(entry.weight)
        except Exception as e:
            # On failure, attempt to clean up any already-loaded LoRAs
            log.error(
                "Failed to load LoRA '%s': %s",
                entry.filename,
                e,
            )
            # Best-effort cleanup of partially applied LoRAs
            try:
                unload_loras(pipeline)
            except Exception:
                pass  # Don't mask the original error

            # Raise with plain-language message identifying the failed LoRA
            # AND the underlying cause (Requirement 2.7)
            raise RuntimeError(_lora_failure_message(entry.filename, e)) from e

    # Set all adapters active with their respective weights
    if adapter_names:
        try:
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        except Exception as e:
            log.error("Failed to set LoRA adapter weights: %s", e)
            try:
                unload_loras(pipeline)
            except Exception:
                pass
            raise RuntimeError(
                f"Could not apply LoRA weights — "
                f"{type(e).__name__}: {e}. "
                f"Try removing some LoRAs and regenerating."
            ) from e

    # Forge Neo equivalent of "Automatic (fp16 LoRA)" for fp8 base models
    _cast_lora_layers_to_compute_dtype(pipeline)

    log.info("All %d LoRA(s) applied successfully", len(stack.entries))


def unload_loras(pipeline: Any) -> None:
    """Remove all LoRA weights from pipeline, restoring base model.

    Called after sampling completes (success or failure) to ensure the
    base model weights are clean for the next generation (Requirement 2.6, 8.2).

    The base checkpoint pipeline remains cached in memory — no disk I/O
    for base model weights occurs (Requirement 10.3).

    Args:
        pipeline: A diffusers pipeline object with unload_lora_weights support.
    """
    try:
        pipeline.unload_lora_weights()
        log.info("LoRA weights unloaded — base model restored")
    except Exception as e:
        # Log but don't crash — the base model should still be usable
        log.warning("Error unloading LoRA weights: %s", e)


# ---------------------------------------------------------------------------
# Stack operations — JSON-based for Gradio UI state passing
# ---------------------------------------------------------------------------


def _stack_from_json(stack_json: str) -> list[dict[str, Any]]:
    """Parse a LoRA stack JSON string into a list of entry dicts.

    Each entry dict has keys: "path" (str) and "weight" (float).
    Returns an empty list if the JSON is empty, null, or invalid.
    """
    if not stack_json or stack_json.strip() in ("", "null", "[]"):
        return []
    try:
        data = json.loads(stack_json)
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, TypeError):
        return []


def _stack_to_json(entries: list[dict[str, Any]]) -> str:
    """Serialize a list of LoRA entry dicts back to JSON."""
    return json.dumps(entries)


def add_lora_to_stack(
    lora_name: str,
    weight: float,
    current_stack_json: str,
    loras_dir: Path | None = None,
) -> tuple[str, str]:
    """Add a LoRA to the stack, checking for duplicates by file path.

    Args:
        lora_name: Display name (stem) of the LoRA file to add.
        weight: Weight multiplier (0.0–2.0) for the LoRA.
        current_stack_json: JSON string of the current LoRA stack.
        loras_dir: Directory containing LoRA files. Defaults to Config.LORAS_DIR.

    Returns:
        A tuple of (updated_stack_json, message).
        - On success: the updated stack JSON and an empty message.
        - On duplicate: the unchanged stack JSON and an informational message.
    """
    if loras_dir is None:
        loras_dir = get_loras_dir()

    entries = _stack_from_json(current_stack_json)

    # Resolve the file path for the new LoRA
    file_path = loras_dir / f"{lora_name}.safetensors"
    file_path_str = str(file_path)

    # Check for duplicate by file path (Requirement 2.3)
    for entry in entries:
        if entry.get("path") == file_path_str:
            message = f"'{lora_name}' is already in the LoRA stack."
            return _stack_to_json(entries), message

    # Add the new entry
    new_entry = {"path": file_path_str, "weight": weight}
    entries.append(new_entry)

    return _stack_to_json(entries), ""


def remove_lora_from_stack(
    lora_name: str,
    current_stack_json: str,
    loras_dir: Path | None = None,
) -> tuple[str, str]:
    """Remove a LoRA from the stack by filename.

    Args:
        lora_name: Display name (stem) of the LoRA file to remove.
        current_stack_json: JSON string of the current LoRA stack.
        loras_dir: Directory containing LoRA files. Defaults to Config.LORAS_DIR.

    Returns:
        A tuple of (updated_stack_json, message).
        - On success: the updated stack JSON and an empty message.
        - If not found: the unchanged stack JSON and an informational message.
    """
    if loras_dir is None:
        loras_dir = get_loras_dir()

    entries = _stack_from_json(current_stack_json)

    # Resolve the file path for the LoRA to remove
    file_path = loras_dir / f"{lora_name}.safetensors"
    file_path_str = str(file_path)

    # Filter out the matching entry
    original_len = len(entries)
    entries = [e for e in entries if e.get("path") != file_path_str]

    if len(entries) == original_len:
        message = f"'{lora_name}' was not found in the LoRA stack."
        return _stack_to_json(entries), message

    return _stack_to_json(entries), ""
