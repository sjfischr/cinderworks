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


def apply_loras(pipeline: Any, stack: LoRAStack, vram_manager: Any = None) -> None:
    """Apply LoRA stack to pipeline in order using diffusers load_lora_weights.

    Each LoRA is loaded fresh and applied with its configured weight.
    The function coordinates with the VRAM manager to pre-check that the
    combined footprint (base model + LoRAs) fits within the budget.

    This function is called AFTER the diffusion model tenant is acquired
    on GPU and BEFORE sampling begins (Requirement 8.1).

    Args:
        pipeline: A diffusers pipeline object with load_lora_weights support.
        stack: The ordered LoRA stack to apply.
        vram_manager: Optional VRAMManager instance for VRAM budget checks.
            If None, the shared app-wide manager is used.

    Raises:
        RuntimeError: If VRAM budget is exceeded or any LoRA fails to load.
            The error message identifies the specific LoRA that failed.
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
        adapter_name = f"lora_{i}_{entry.filename}"
        log.info(
            "Loading LoRA %d/%d: '%s' (weight=%.2f)",
            i + 1,
            len(stack.entries),
            entry.filename,
            entry.weight,
        )

        try:
            pipeline.load_lora_weights(
                str(entry.file_path),
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
            # (Requirement 2.7)
            raise RuntimeError(
                f"Could not load LoRA '{entry.filename}' — "
                f"the file may be corrupted or incompatible. "
                f"Remove it from the stack and try again."
            ) from e

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
                f"the combination may be incompatible. "
                f"Try removing some LoRAs and regenerating."
            ) from e

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
