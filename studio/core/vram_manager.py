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

# Minimum reserve subtracted from total VRAM when computing usable
# capacity. On Windows/WDDM, the desktop compositor, browser, and driver
# reserve ~1-1.5 GB of dedicated VRAM. A tenant estimate that exceeds
# usable (not total) capacity will spill into shared system memory via
# WDDM sysmem fallback — the GPU shows high utilization while the copy
# engine thrashes weights over PCIe and generation slows to a crawl.
# Refusing up front (R6.4, R7.5) is the honest failure mode.
_VRAM_RESERVE = 1_500_000_000  # ~1.5 GB floor

# Safety margin added on top of the memory other processes are actually
# holding at detection time. Desktop usage isn't static — browsers and
# the compositor grow and shrink — so budgeting right up to the current
# free amount invites a spill the moment anything else allocates.
_VRAM_SAFETY = 1_000_000_000  # ~1 GB


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

        # Check if we can fit the new tenant against the usable budget.
        # self._total_vram IS the usable budget: an explicitly passed value
        # is taken at face value, and auto-detection already subtracts the
        # WDDM/desktop reserve (see _detect_total_vram). After releasing
        # the prior resident, PyTorch's allocator may still hold reserved
        # memory that mem_get_info doesn't report as free, so we budget
        # against capacity rather than instantaneous free memory.
        #
        # There is deliberately NO grace band above the budget. An
        # over-capacity load doesn't OOM on Windows; it silently spills
        # into shared system memory via WDDM sysmem fallback and thrashes
        # the copy engine. Refusal here is the protective behavior
        # (R6.4, R7.5).
        usable = self._total_vram
        log.info(
            "VRAM check for '%s': needs %s, usable VRAM budget %s, torch free %s",
            tenant.name,
            _format_bytes(tenant.estimated_bytes),
            _format_bytes(usable),
            _format_bytes(self._get_torch_free()),
        )
        if tenant.estimated_bytes > usable:
            log.error(
                "VRAM insufficient for '%s': needs %s but usable budget is %s",
                tenant.name,
                _format_bytes(tenant.estimated_bytes),
                _format_bytes(usable),
            )
            raise VRAMError(
                f"Not enough VRAM — '{tenant.name}' needs "
                f"{_format_bytes(tenant.estimated_bytes)} but only "
                f"{_format_bytes(usable)} is usable on this card. "
                f"Try lowering batch size or switching to fp8_scaled precision."
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
        """Detect the usable VRAM budget via torch.cuda, or fall back.

        The reserve is dynamic: whatever OTHER processes (desktop
        compositor, browser, driver) are holding at detection time plus
        a 1 GB safety margin, floored at the 1.5 GB static estimate.
        Torch's own reserved pool is explicitly EXCLUDED from the
        "others" figure — it is our cache (the resident pipeline between
        generations) and is fully reusable by the next tenant. Counting
        it as foreign collapsed the budget to ~1 GB on every generation
        after the first. An explicitly passed total_vram (tests, callers
        with their own accounting) bypasses this and is taken at face
        value.
        """
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                ours = torch.cuda.memory_reserved()
                in_use_by_others = max(0, total - free - ours)
                reserve = max(_VRAM_RESERVE, in_use_by_others + _VRAM_SAFETY)
                usable = total - reserve
                log.info(
                    "Detected total VRAM: %s (%s in use by other processes, "
                    "%s is our own torch cache; reserving %s; usable "
                    "budget %s)",
                    _format_bytes(total),
                    _format_bytes(in_use_by_others),
                    _format_bytes(ours),
                    _format_bytes(reserve),
                    _format_bytes(usable),
                )
                return usable
        except (ImportError, RuntimeError):
            pass
        usable = _DEFAULT_TOTAL_VRAM - _VRAM_RESERVE
        log.info("Using default usable VRAM budget: %s", _format_bytes(usable))
        return usable

    @staticmethod
    def _get_torch_free() -> int:
        """Get torch-reported free VRAM, or -1 if unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return free
        except (ImportError, RuntimeError):
            pass
        return -1


def _format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    if size < 0:
        return "N/A"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.0f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"


# ---------------------------------------------------------------------------
# App-wide singleton
# ---------------------------------------------------------------------------

# The design doc names vram_manager THE single GPU chokepoint — that only
# holds if every generation shares one instance. A fresh instance per
# generation re-detects the budget mid-session, at which point our own
# resident cache distorts the numbers and tenant state is forgotten.
_default_manager: VRAMManager | None = None


def get_vram_manager() -> VRAMManager:
    """Return the shared app-wide VRAMManager, creating it on first use."""
    global _default_manager
    if _default_manager is None:
        _default_manager = VRAMManager()
    return _default_manager


def _reset_vram_manager() -> None:
    """Drop the shared manager. For testing only."""
    global _default_manager
    _default_manager = None
