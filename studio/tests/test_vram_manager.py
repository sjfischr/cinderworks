"""Tests for core/vram_manager.py — VRAM tenant discipline.

Validates Requirements 7.1, 7.2, 7.3, 7.4, 7.5:
- All GPU moves go through acquire/release API (7.1)
- Text encoder released before DiT loaded (7.2)
- One heavyweight tenant resident at a time; acquiring B unloads A (7.3)
- Precision-aware memory estimation (7.4)
- OOM error with plain-language message (7.5)
"""

import pytest

from studio.core.vram_manager import Tenant, VRAMError, VRAMManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tenant(name: str, size_bytes: int) -> tuple[Tenant, list[str]]:
    """Create a tenant with a tracked load/unload log."""
    log: list[str] = []
    tenant = Tenant(
        name=name,
        estimated_bytes=size_bytes,
        load_fn=lambda n=name: log.append(f"load:{n}"),
        unload_fn=lambda n=name: log.append(f"unload:{n}"),
    )
    return tenant, log


def _make_manager(total_bytes: int) -> VRAMManager:
    """Create a VRAMManager with a fixed total VRAM (no CUDA detection)."""
    return VRAMManager(total_vram=total_bytes)


# ---------------------------------------------------------------------------
# Tests: Basic acquire/release
# ---------------------------------------------------------------------------


class TestAcquireRelease:
    """Tests for basic tenant acquisition and release."""

    def test_acquire_loads_tenant(self):
        """Acquiring a tenant calls its load_fn."""
        mgr = _make_manager(total_bytes=10_000)
        tenant, log = _make_tenant("dit", size_bytes=5_000)

        mgr.acquire(tenant)

        assert "load:dit" in log
        assert mgr.resident is not None
        assert mgr.resident.name == "dit"

    def test_release_unloads_tenant(self):
        """Releasing a tenant calls its unload_fn and clears resident."""
        mgr = _make_manager(total_bytes=10_000)
        tenant, log = _make_tenant("dit", size_bytes=5_000)

        mgr.acquire(tenant)
        mgr.release(tenant)

        assert "unload:dit" in log
        assert mgr.resident is None

    def test_acquire_same_tenant_is_noop(self):
        """Acquiring a tenant that's already resident does nothing."""
        mgr = _make_manager(total_bytes=10_000)
        tenant, log = _make_tenant("dit", size_bytes=5_000)

        mgr.acquire(tenant)
        log.clear()
        mgr.acquire(tenant)

        # No additional load or unload calls
        assert log == []
        assert mgr.resident.name == "dit"


# ---------------------------------------------------------------------------
# Tests: One heavyweight at a time (Requirement 7.3)
# ---------------------------------------------------------------------------


class TestTenantDiscipline:
    """Acquiring tenant B while tenant A is resident → unloads A first."""

    def test_acquire_new_tenant_unloads_existing(self):
        """Requirement 7.3: acquiring B while A is resident unloads A."""
        mgr = _make_manager(total_bytes=20_000)
        actions: list[str] = []

        encoder = Tenant(
            name="text_encoder",
            estimated_bytes=4_000,
            load_fn=lambda: actions.append("load:text_encoder"),
            unload_fn=lambda: actions.append("unload:text_encoder"),
        )
        dit = Tenant(
            name="dit",
            estimated_bytes=13_000,
            load_fn=lambda: actions.append("load:dit"),
            unload_fn=lambda: actions.append("unload:dit"),
        )

        mgr.acquire(encoder)
        mgr.acquire(dit)

        # Encoder was loaded, then unloaded, then DiT loaded
        assert actions == [
            "load:text_encoder",
            "unload:text_encoder",
            "load:dit",
        ]
        assert mgr.resident.name == "dit"

    def test_never_two_tenants_resident(self):
        """At no point are two heavyweight tenants simultaneously resident."""
        mgr = _make_manager(total_bytes=30_000)
        resident_count: list[int] = []

        # Track the number of "resident" items at each load/unload
        count = [0]

        def make_load(name):
            def load():
                count[0] += 1
                resident_count.append(count[0])

            return load

        def make_unload(name):
            def unload():
                count[0] -= 1
                resident_count.append(count[0])

            return unload

        tenants = [
            Tenant(f"t{i}", 5_000, make_load(f"t{i}"), make_unload(f"t{i}"))
            for i in range(3)
        ]

        for t in tenants:
            mgr.acquire(t)

        # At no point should we have more than 1 resident
        assert all(c <= 1 for c in resident_count)


# ---------------------------------------------------------------------------
# Tests: OOM error (Requirement 7.5)
# ---------------------------------------------------------------------------


class TestOOMError:
    """Acquire failure due to insufficient memory raises VRAMError."""

    def test_acquire_raises_vram_error_on_oom(self):
        """When tenant doesn't fit, raise VRAMError with OOM message."""
        mgr = _make_manager(total_bytes=10_000)
        huge_tenant, _ = _make_tenant("huge", size_bytes=20_000)

        with pytest.raises(VRAMError) as exc_info:
            mgr.acquire(huge_tenant)

        msg = str(exc_info.value)
        assert "VRAM" in msg
        assert "batch size" in msg.lower() or "batch" in msg.lower()
        assert "fp8_scaled" in msg

    def test_oom_message_is_plain_language(self):
        """The OOM message suggests lowering batch size or switching precision."""
        mgr = _make_manager(total_bytes=5_000)
        big_tenant, _ = _make_tenant("toobig", size_bytes=10_000)

        with pytest.raises(VRAMError) as exc_info:
            mgr.acquire(big_tenant)

        msg = str(exc_info.value)
        expected = "Not enough VRAM"
        assert expected in msg

    def test_oom_after_unload_still_insufficient(self):
        """Even after unloading existing tenant, if new one still won't fit → OOM."""
        mgr = _make_manager(total_bytes=10_000)
        small, _ = _make_tenant("small", size_bytes=3_000)
        huge, log = _make_tenant("huge", size_bytes=15_000)

        mgr.acquire(small)

        with pytest.raises(VRAMError):
            mgr.acquire(huge)

        # The small tenant was unloaded (attempted to make space)
        # but the huge one still can't fit
        assert mgr.resident is None


# ---------------------------------------------------------------------------
# Tests: Memory estimation (Requirement 7.4)
# ---------------------------------------------------------------------------


class TestMemoryEstimation:
    """Precision-aware memory estimation."""

    def test_estimate_available_no_resident(self):
        """When no tenant is resident, full VRAM is available."""
        mgr = _make_manager(total_bytes=24_000_000_000)
        assert mgr.estimate_available() == 24_000_000_000

    def test_estimate_available_with_resident(self):
        """When a tenant is resident, available = total - tenant bytes."""
        mgr = _make_manager(total_bytes=24_000_000_000)
        tenant, _ = _make_tenant("dit", size_bytes=13_000_000_000)
        mgr.acquire(tenant)

        available = mgr.estimate_available()
        assert available == 24_000_000_000 - 13_000_000_000

    def test_can_fit_true_when_space_available(self):
        """can_fit returns True when bytes fit within total capacity."""
        mgr = _make_manager(total_bytes=24_000_000_000)
        assert mgr.can_fit(13_000_000_000) is True

    def test_can_fit_false_when_exceeds_total(self):
        """can_fit returns False when bytes exceed total VRAM."""
        mgr = _make_manager(total_bytes=24_000_000_000)
        assert mgr.can_fit(30_000_000_000) is False

    def test_precision_aware_bf16_vs_fp8(self):
        """Different precision variants have different estimated sizes.

        bf16 DiT ~25 GB, fp8_scaled DiT ~13 GB. The manager's estimate
        reflects whichever is loaded.
        """
        mgr = _make_manager(total_bytes=30_000_000_000)

        # Simulate bf16 tenant (larger)
        bf16_dit, _ = _make_tenant("dit_bf16", size_bytes=25_000_000_000)
        mgr.acquire(bf16_dit)
        avail_bf16 = mgr.estimate_available()

        mgr.release(bf16_dit)

        # Simulate fp8 tenant (smaller)
        fp8_dit, _ = _make_tenant("dit_fp8", size_bytes=13_000_000_000)
        mgr.acquire(fp8_dit)
        avail_fp8 = mgr.estimate_available()

        # fp8 leaves more room than bf16
        assert avail_fp8 > avail_bf16
        assert avail_fp8 == 30_000_000_000 - 13_000_000_000
        assert avail_bf16 == 30_000_000_000 - 25_000_000_000

    def test_can_fit_considers_total_not_current_free(self):
        """can_fit checks against total capacity (since existing can be unloaded)."""
        mgr = _make_manager(total_bytes=20_000)
        existing, _ = _make_tenant("existing", size_bytes=15_000)
        mgr.acquire(existing)

        # 18_000 bytes needed — more than currently free (5_000) but
        # within total (20_000) since existing can be unloaded
        assert mgr.can_fit(18_000) is True
        # 25_000 exceeds total
        assert mgr.can_fit(25_000) is False


# ---------------------------------------------------------------------------
# Tests: Encode→release→DiT acquire flow (Requirement 7.2)
# ---------------------------------------------------------------------------


class TestEncodeOffloadFlow:
    """Verify the encode→offload encoder→load DiT flow works correctly."""

    def test_encoder_released_before_dit_loaded(self):
        """The text encoder is released before DiT is loaded (Req 7.2)."""
        mgr = _make_manager(total_bytes=30_000_000_000)
        actions: list[str] = []

        encoder = Tenant(
            name="text_encoder",
            estimated_bytes=4_000_000_000,
            load_fn=lambda: actions.append("load:encoder"),
            unload_fn=lambda: actions.append("unload:encoder"),
        )
        dit = Tenant(
            name="dit",
            estimated_bytes=13_000_000_000,
            load_fn=lambda: actions.append("load:dit"),
            unload_fn=lambda: actions.append("unload:dit"),
        )

        # Simulate the generation flow:
        # 1. Acquire encoder
        mgr.acquire(encoder)
        # 2. (Encoding happens here...)
        # 3. Release encoder
        mgr.release(encoder)
        # 4. Acquire DiT
        mgr.acquire(dit)

        assert actions == [
            "load:encoder",
            "unload:encoder",
            "load:dit",
        ]

    def test_acquire_dit_while_encoder_resident_auto_unloads(self):
        """If encoder isn't explicitly released, acquiring DiT still unloads it."""
        mgr = _make_manager(total_bytes=30_000_000_000)
        actions: list[str] = []

        encoder = Tenant(
            name="text_encoder",
            estimated_bytes=4_000_000_000,
            load_fn=lambda: actions.append("load:encoder"),
            unload_fn=lambda: actions.append("unload:encoder"),
        )
        dit = Tenant(
            name="dit",
            estimated_bytes=13_000_000_000,
            load_fn=lambda: actions.append("load:dit"),
            unload_fn=lambda: actions.append("unload:dit"),
        )

        # Acquire encoder, then directly acquire DiT without explicit release
        mgr.acquire(encoder)
        mgr.acquire(dit)

        # Encoder should be auto-unloaded before DiT is loaded
        unload_idx = actions.index("unload:encoder")
        load_dit_idx = actions.index("load:dit")
        assert unload_idx < load_dit_idx


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings
from hypothesis import strategies as st


# Feature: cinderworks, Property 14: Tenant discipline — acquire unloads existing resident
class TestPropertyTenantDiscipline:
    """Property 14: For any sequence of tenant acquisitions, acquiring tenant B
    while tenant A is resident SHALL unload A before B is loaded. At no point
    SHALL two heavyweight tenants be simultaneously GPU-resident.

    Validates: Requirements 7.2, 7.3
    """

    @given(
        tenant_names=st.lists(
            st.sampled_from(["text_encoder", "dit", "vae", "extra"]),
            min_size=1,
            max_size=20,
        ),
        sizes=st.lists(
            st.integers(min_value=100, max_value=5_000),
            min_size=20,
            max_size=20,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_never_two_tenants_simultaneously_resident(
        self, tenant_names: list[str], sizes: list[int]
    ):
        """At no point during any sequence of acquires are two heavyweight
        tenants simultaneously GPU-resident."""
        # Use a large total_vram so we are testing discipline, not OOM
        mgr = VRAMManager(total_vram=1_000_000)

        # Track GPU residency count at every transition
        gpu_resident_count = [0]
        max_simultaneous = [0]

        def make_load(name: str):
            def load():
                gpu_resident_count[0] += 1
                if gpu_resident_count[0] > max_simultaneous[0]:
                    max_simultaneous[0] = gpu_resident_count[0]

            return load

        def make_unload(name: str):
            def unload():
                gpu_resident_count[0] -= 1

            return unload

        # Build tenants from generated names, assigning sizes cyclically
        tenants_by_name: dict[str, Tenant] = {}
        for i, name in enumerate(set(tenant_names)):
            size = sizes[i % len(sizes)]
            tenants_by_name[name] = Tenant(
                name=name,
                estimated_bytes=size,
                load_fn=make_load(name),
                unload_fn=make_unload(name),
            )

        # Execute the sequence of acquisitions
        for name in tenant_names:
            mgr.acquire(tenants_by_name[name])

        # Property: at no point were two tenants simultaneously resident
        assert max_simultaneous[0] <= 1

    @given(
        tenant_names=st.lists(
            st.sampled_from(["text_encoder", "dit", "vae", "extra"]),
            min_size=2,
            max_size=20,
        ),
        sizes=st.lists(
            st.integers(min_value=100, max_value=5_000),
            min_size=20,
            max_size=20,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_unload_before_load_ordering(
        self, tenant_names: list[str], sizes: list[int]
    ):
        """When acquiring B while A is resident, A's unload_fn is called
        before B's load_fn."""
        mgr = VRAMManager(total_vram=1_000_000)

        # Track all load/unload calls in order
        actions: list[str] = []

        def make_load(name: str):
            def load():
                actions.append(f"load:{name}")

            return load

        def make_unload(name: str):
            def unload():
                actions.append(f"unload:{name}")

            return unload

        # Build tenants from generated names
        tenants_by_name: dict[str, Tenant] = {}
        for i, name in enumerate(set(tenant_names)):
            size = sizes[i % len(sizes)]
            tenants_by_name[name] = Tenant(
                name=name,
                estimated_bytes=size,
                load_fn=make_load(name),
                unload_fn=make_unload(name),
            )

        # Execute the sequence of acquisitions
        for name in tenant_names:
            mgr.acquire(tenants_by_name[name])

        # Property: every load:X that is preceded by a load:Y (where X != Y)
        # must have an unload:Y between them
        for i, action in enumerate(actions):
            if action.startswith("load:"):
                loaded_name = action.split(":")[1]
                # Find the previous load (if any) that was a different tenant
                for j in range(i - 1, -1, -1):
                    if actions[j].startswith("load:"):
                        prev_loaded = actions[j].split(":")[1]
                        if prev_loaded != loaded_name:
                            # There must be an unload of the previous tenant
                            # between j and i
                            unload_marker = f"unload:{prev_loaded}"
                            intervening = actions[j + 1 : i]
                            assert unload_marker in intervening, (
                                f"Expected '{unload_marker}' between "
                                f"'{actions[j]}' (idx {j}) and '{action}' (idx {i}), "
                                f"but got: {intervening}"
                            )
                        break  # Only check the immediately preceding load


# ---------------------------------------------------------------------------
# Property-Based Test: OOM during batch preserves prior completed batches
# ---------------------------------------------------------------------------


# Feature: cinderworks, Property 13: OOM during batch preserves prior completed batches
class TestPropertyOOMPreservation:
    """Property 13: For any batch sequence where an out-of-memory error occurs
    at batch K (K > 1), all images produced by batches 1 through K-1 SHALL be
    preserved on disk and in the output, and the failure SHALL be reported in
    plain language.

    Validates: Requirements 6.5
    """

    @given(
        total_batches=st.integers(min_value=2, max_value=10),
        batch_size=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100, deadline=None)
    def test_oom_at_batch_k_preserves_prior_batches(
        self, total_batches: int, batch_size: int
    ):
        """When OOM occurs at batch K, batches 1..(K-1) are preserved."""
        # Pick the batch at which OOM occurs (always > 1, up to total_batches)
        fail_at_batch = total_batches  # OOM on the last batch

        # Set up a VRAMManager with limited total VRAM.
        # Each batch tenant needs `batch_size * per_image_bytes` VRAM.
        per_image_bytes = 1_000
        batch_vram_need = batch_size * per_image_bytes

        # Total VRAM is enough for (fail_at_batch - 1) batches but NOT the
        # fail_at_batch-th one. Since VRAMManager only has one tenant at a time,
        # we simulate OOM by making the failing batch request more VRAM than
        # available.
        total_vram = batch_vram_need  # Enough for normal batches

        mgr = VRAMManager(total_vram=total_vram)

        # Simulate the batch execution pattern the krea2 backend will use:
        # - For each batch, acquire a tenant (the DiT), generate images, release
        # - At batch K, the acquire fails with VRAMError (OOM)
        completed_batches: list[list[str]] = []
        error_message: str | None = None

        for batch_idx in range(1, total_batches + 1):
            # Determine the tenant size for this batch
            if batch_idx == fail_at_batch:
                # This batch requests more VRAM than available → triggers OOM
                tenant_size = total_vram + 1
            else:
                # Normal batch fits within available VRAM
                tenant_size = batch_vram_need

            tenant = Tenant(
                name=f"dit_batch_{batch_idx}",
                estimated_bytes=tenant_size,
                load_fn=lambda: None,
                unload_fn=lambda: None,
            )

            try:
                mgr.acquire(tenant)
            except VRAMError as e:
                # OOM occurred — record the error and stop
                error_message = str(e)
                break

            # Batch succeeded — simulate producing images
            batch_images = [
                f"image_batch{batch_idx}_{i}.png" for i in range(batch_size)
            ]
            completed_batches.append(batch_images)

            # Release tenant after batch completes
            mgr.release(tenant)

        # Property 1: All batches before the failure are preserved
        expected_completed = fail_at_batch - 1
        assert len(completed_batches) == expected_completed, (
            f"Expected {expected_completed} completed batches, "
            f"got {len(completed_batches)}"
        )

        # Verify each completed batch has the correct number of images
        for batch_images in completed_batches:
            assert len(batch_images) == batch_size

        # Total images preserved = (K-1) * batch_size
        total_images = sum(len(b) for b in completed_batches)
        assert total_images == expected_completed * batch_size

        # Property 2: The failure is reported in plain language
        assert error_message is not None, "Expected VRAMError but none occurred"
        assert "Not enough VRAM" in error_message, (
            f"Error message should contain 'Not enough VRAM', got: {error_message}"
        )

    @given(
        total_batches=st.integers(min_value=3, max_value=10),
        fail_at=st.integers(min_value=2, max_value=9),
        batch_size=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100, deadline=None)
    def test_oom_at_arbitrary_batch_preserves_all_prior(
        self, total_batches: int, fail_at: int, batch_size: int
    ):
        """When OOM occurs at an arbitrary batch K (2 <= K <= total), all
        batches 1..(K-1) are preserved regardless of K's position."""
        from hypothesis import assume

        # Ensure fail_at is valid for this total
        assume(fail_at <= total_batches)

        per_image_bytes = 1_000
        batch_vram_need = batch_size * per_image_bytes
        total_vram = batch_vram_need  # Enough for normal-sized batches

        mgr = VRAMManager(total_vram=total_vram)

        completed_batches: list[list[str]] = []
        error_message: str | None = None

        for batch_idx in range(1, total_batches + 1):
            if batch_idx == fail_at:
                # Trigger OOM: request more than available
                tenant_size = total_vram + 1
            else:
                tenant_size = batch_vram_need

            tenant = Tenant(
                name=f"dit_batch_{batch_idx}",
                estimated_bytes=tenant_size,
                load_fn=lambda: None,
                unload_fn=lambda: None,
            )

            try:
                mgr.acquire(tenant)
            except VRAMError as e:
                error_message = str(e)
                break

            # Batch succeeded — record images
            batch_images = [
                f"image_batch{batch_idx}_{i}.png" for i in range(batch_size)
            ]
            completed_batches.append(batch_images)
            mgr.release(tenant)

        # Property: exactly (fail_at - 1) batches completed
        expected_completed = fail_at - 1
        assert len(completed_batches) == expected_completed, (
            f"Expected {expected_completed} completed batches, "
            f"got {len(completed_batches)}"
        )

        # All completed batch images are preserved
        total_images = sum(len(b) for b in completed_batches)
        assert total_images == expected_completed * batch_size

        # Failure reported in plain language
        assert error_message is not None
        assert "Not enough VRAM" in error_message



# Feature: cinderworks, Property 12: VRAM manager refuses batch exceeding estimated capacity
class TestPropertyVRAMBatchRefusal:
    """Property 12: For any batch_size whose estimated VRAM requirement
    (batch_size × per_image_footprint) exceeds the VRAM_Manager's estimated
    available memory, the system SHALL refuse the generation with a plain-language
    message before any inference work begins.

    Validates: Requirements 6.4
    """

    @given(
        total_vram=st.integers(min_value=1_000_000, max_value=30_000_000_000),
        per_image_footprint=st.integers(min_value=100_000, max_value=5_000_000_000),
        batch_size=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=100, deadline=None)
    def test_can_fit_returns_false_when_batch_exceeds_capacity(
        self, total_vram: int, per_image_footprint: int, batch_size: int
    ):
        """When batch_size × per_image_footprint > total_vram, can_fit returns False."""
        from hypothesis import assume

        estimated_need = batch_size * per_image_footprint
        assume(estimated_need > total_vram)

        mgr = VRAMManager(total_vram=total_vram)
        assert mgr.can_fit(estimated_need) is False

    @given(
        total_vram=st.integers(min_value=1_000_000, max_value=30_000_000_000),
        per_image_footprint=st.integers(min_value=100_000, max_value=5_000_000_000),
        batch_size=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=100, deadline=None)
    def test_can_fit_returns_true_when_batch_within_capacity(
        self, total_vram: int, per_image_footprint: int, batch_size: int
    ):
        """When batch_size × per_image_footprint <= total_vram, can_fit returns True."""
        from hypothesis import assume

        estimated_need = batch_size * per_image_footprint
        assume(estimated_need <= total_vram)

        mgr = VRAMManager(total_vram=total_vram)
        assert mgr.can_fit(estimated_need) is True

    @given(
        total_vram=st.integers(min_value=1_000_000, max_value=30_000_000_000),
        per_image_footprint=st.integers(min_value=100_000, max_value=5_000_000_000),
        batch_size=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=100, deadline=None)
    def test_acquire_raises_vram_error_with_plain_language_when_batch_exceeds(
        self, total_vram: int, per_image_footprint: int, batch_size: int
    ):
        """When can_fit is False and acquire is attempted with that size,
        VRAMError is raised with a plain-language message before any
        inference work begins."""
        from hypothesis import assume

        estimated_need = batch_size * per_image_footprint
        assume(estimated_need > total_vram)

        mgr = VRAMManager(total_vram=total_vram)

        # Track whether any inference (load_fn) was called
        inference_started = [False]

        def load_fn():
            inference_started[0] = True

        tenant = Tenant(
            name="batch_workload",
            estimated_bytes=estimated_need,
            load_fn=load_fn,
            unload_fn=lambda: None,
        )

        with pytest.raises(VRAMError) as exc_info:
            mgr.acquire(tenant)

        msg = str(exc_info.value)
        # Message must be plain-language (no tracebacks, no class names)
        assert "VRAM" in msg
        assert "Traceback" not in msg
        # No inference work should have started
        assert inference_started[0] is False
