"""VRAM Tenant Manager — the single GPU chokepoint.

Only this module moves anything on/off the GPU. Nothing else calls
.to('cuda') / .to('cpu') directly. Every GPU consumer (text encoder, DiT,
and later the prompt LLM and trainer) registers as a Tenant.

Phase 1 enforces: at most one heavyweight tenant resident at a time.
Acquiring tenant B while tenant A is resident → unloads A first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)


class VRAMError(Exception):
    """Raised when a GPU memory operation fails due to insufficient VRAM."""

    pass


@dataclass
class Tenant:
    """A component that wants GPU residency.

    Phase 1 tenants: 'text_encoder' and 'dit'.
    The interface is designed so Phase 3/4 tenants (prompt LLM, trainer)
    slot in unchanged.
    """

    name: str  # 'text_encoder', 'dit'
    estimated_bytes: int  # VRAM footprint in bytes
    load_fn: Callable[[], None]  # called on acquire (moves to GPU)
    unload_fn: Callable[[], None]  # called on release (moves to CPU)


# Default total VRAM fallback when CUDA is not available (24 GB, RTX 4090)
_DEFAULT_TOTAL_VRAM = 24 * 1024 * 1024 * 1024  # 24 GiB


class VRAMManager:
    """Central coordinator for all GPU memory allocation.

    Enforces tenant discipline: one heavyweight model resident at a time.
    All GPU moves go through acquire/release — nothing else touches CUDA
    device placement directly.
    """

    def __init__(self, total_vram: int | None = None) -> None:
        """Initialize the VRAM manager.

        Args:
            total_vram: Total available VRAM in bytes. If None, queries
                torch.cuda.mem_get_info() when CUDA is available, otherwise
                falls back to _DEFAULT_TOTAL_VRAM.
        """
        self._total_vram = total_vram if total_vram is not None else self._detect_total_vram()
        self._resident: Tenant | None = None

    def acquire(self, tenant: Tenant) -> None:
        """Load a tenant to GPU. Unloads any existing resident first.

        If another heavyweight tenant is currently resident, it is released
        before the new tenant is loaded. If there is insufficient memory
        even after releasing the existing tenant, raises VRAMError with a
        plain-language message.

        Args:
            tenant: The Tenant to load onto the GPU.

        Raises:
            VRAMError: If insufficient VRAM after releasing existing tenants.
        """
        # If the same tenant is already resident, nothing to do
        if self._resident is not None and self._resident.name == tenant.name:
            log.debug("Tenant '%s' already resident, skipping acquire", tenant.name)
            return

        # Unload current resident if one exists (one heavyweight at a time)
        if self._resident is not None:
            log.info(
                "Unloading tenant '%s' to make room for '%s'",
                self._resident.name,
                tenant.name,
            )
            self._do_release(self._resident)

        # Check if we can fit the new tenant
        available = self.estimate_available()
        if tenant.estimated_bytes > available:
            raise VRAMError(
                "Not enough VRAM \u2014 try lowering batch size or switching to fp8_scaled precision"
            )

        # Load the new tenant
        log.info(
            "Acquiring tenant '%s' (%d bytes)",
            tenant.name,
            tenant.estimated_bytes,
        )
        tenant.load_fn()
        self._resident = tenant

    def release(self, tenant: Tenant) -> None:
        """Move a tenant back to CPU, freeing GPU memory.

        Args:
            tenant: The Tenant to release from GPU.
        """
        if self._resident is not None and self._resident.name == tenant.name:
            self._do_release(tenant)
        else:
            log.warning(
                "Attempted to release tenant '%s' but it is not the current resident",
                tenant.name,
            )

    def estimate_available(self) -> int:
        """Estimated free VRAM in bytes.

        Uses torch.cuda.mem_get_info() when CUDA is available, otherwise
        computes based on total minus resident tenant footprint.
        """
        try:
            import torch

            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return free
        except (ImportError, RuntimeError):
            pass

        # Fallback: total minus current resident's footprint
        used = self._resident.estimated_bytes if self._resident is not None else 0
        return self._total_vram - used

    def can_fit(self, bytes_needed: int) -> bool:
        """Pre-check whether a given number of bytes would fit in VRAM.

        This accounts for the possibility of releasing the current resident
        first — i.e., it checks against the total capacity, not just
        currently free memory.

        Args:
            bytes_needed: The number of bytes that need to fit.

        Returns:
            True if the bytes can fit (possibly after releasing current resident).
        """
        return bytes_needed <= self._total_vram

    @property
    def resident(self) -> Tenant | None:
        """The currently GPU-resident tenant, or None."""
        return self._resident

    def _do_release(self, tenant: Tenant) -> None:
        """Internal release: call unload and clear resident."""
        log.info("Releasing tenant '%s'", tenant.name)
        tenant.unload_fn()
        self._resident = None

    @staticmethod
    def _detect_total_vram() -> int:
        """Detect total VRAM via torch.cuda, or fall back to default."""
        try:
            import torch

            if torch.cuda.is_available():
                _, total = torch.cuda.mem_get_info()
                return total
        except (ImportError, RuntimeError):
            pass
        return _DEFAULT_TOTAL_VRAM
