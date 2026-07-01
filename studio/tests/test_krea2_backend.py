"""Property-based tests for models/backends/krea2.py — Krea 2 Turbo Backend.

Validates Requirements 5.2, 5.4, 5.5, 6.1, 6.2:
- Sampler params default to Turbo unless explicitly overridden (5.2)
- Seed determinism: explicit seed used as-is, None → random generated (5.4)
- Parameter bounds validation: in-bounds accepted, out-of-bounds rejected (5.5)
- Batch produces correct image count with correct per-image seeds (6.1, 6.2)

All tests are pure logic — NO network, NO model weights, NO CUDA.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from studio.models.backends.krea2 import (
    TURBO_STEPS,
    TURBO_CFG,
    TURBO_MU_SHIFT,
    STEPS_MIN,
    STEPS_MAX,
    SEED_MIN,
    SEED_MAX,
    WIDTH_MIN,
    WIDTH_MAX,
    HEIGHT_MIN,
    HEIGHT_MAX,
    SIZE_MULTIPLE,
    BATCH_SIZE_MIN,
    BATCH_SIZE_MAX,
    BATCH_COUNT_MIN,
    BATCH_COUNT_MAX,
    validate_params,
    resolve_seed,
    generate,
)
from studio.core.vram_manager import VRAMManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid parameter strategies
valid_steps = st.integers(min_value=STEPS_MIN, max_value=STEPS_MAX)
valid_seed = st.integers(min_value=SEED_MIN, max_value=SEED_MAX)
valid_width = st.integers(min_value=WIDTH_MIN // SIZE_MULTIPLE, max_value=WIDTH_MAX // SIZE_MULTIPLE).map(
    lambda x: x * SIZE_MULTIPLE
)
valid_height = st.integers(min_value=HEIGHT_MIN // SIZE_MULTIPLE, max_value=HEIGHT_MAX // SIZE_MULTIPLE).map(
    lambda x: x * SIZE_MULTIPLE
)
valid_batch_size = st.integers(min_value=BATCH_SIZE_MIN, max_value=BATCH_SIZE_MAX)
valid_batch_count = st.integers(min_value=BATCH_COUNT_MIN, max_value=BATCH_COUNT_MAX)
valid_precision = st.sampled_from(["bf16", "fp8_scaled"])
valid_cfg = st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
valid_mu_shift = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)

# A non-empty prompt strategy
non_empty_prompt = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Subsets of param keys that can be omitted
SAMPLER_PARAM_KEYS = ["steps", "cfg", "mu_shift"]


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 8: Sampler parameters default to Turbo
#          unless explicitly overridden
# ---------------------------------------------------------------------------


class TestPropertySamplerDefaults:
    """Property 8: For any subset of sampler parameters provided by the user,
    omitted parameters SHALL use Turbo_Defaults (steps=8, cfg=1.0, mu_shift=1.15),
    and provided parameters SHALL use the user-supplied values exactly.

    **Validates: Requirements 5.2**
    """

    @given(
        prompt=non_empty_prompt,
        include_steps=st.booleans(),
        include_cfg=st.booleans(),
        include_mu_shift=st.booleans(),
        steps=valid_steps,
        cfg=valid_cfg,
        mu_shift=valid_mu_shift,
    )
    @settings(max_examples=100, deadline=None)
    def test_omitted_params_use_turbo_defaults(
        self,
        prompt: str,
        include_steps: bool,
        include_cfg: bool,
        include_mu_shift: bool,
        steps: int,
        cfg: float,
        mu_shift: float,
    ):
        """Omitted sampler params get Turbo defaults; provided params use
        user-supplied values exactly."""
        params: dict = {"prompt": prompt}

        if include_steps:
            params["steps"] = steps
        if include_cfg:
            params["cfg"] = cfg
        if include_mu_shift:
            params["mu_shift"] = mu_shift

        result = validate_params(params)

        # Check steps
        if include_steps:
            assert result.steps == steps, (
                f"Provided steps={steps} but got {result.steps}"
            )
        else:
            assert result.steps == TURBO_STEPS, (
                f"Omitted steps should be {TURBO_STEPS} but got {result.steps}"
            )

        # Check cfg
        if include_cfg:
            assert result.cfg == cfg, (
                f"Provided cfg={cfg} but got {result.cfg}"
            )
        else:
            assert result.cfg == TURBO_CFG, (
                f"Omitted cfg should be {TURBO_CFG} but got {result.cfg}"
            )

        # Check mu_shift
        if include_mu_shift:
            assert result.mu_shift == mu_shift, (
                f"Provided mu_shift={mu_shift} but got {result.mu_shift}"
            )
        else:
            assert result.mu_shift == TURBO_MU_SHIFT, (
                f"Omitted mu_shift should be {TURBO_MU_SHIFT} but got {result.mu_shift}"
            )

    @given(prompt=non_empty_prompt)
    @settings(max_examples=100, deadline=None)
    def test_all_omitted_uses_all_turbo_defaults(self, prompt: str):
        """When all sampler params are omitted, all get Turbo defaults."""
        result = validate_params({"prompt": prompt})

        assert result.steps == TURBO_STEPS
        assert result.cfg == TURBO_CFG
        assert result.mu_shift == TURBO_MU_SHIFT

    @given(
        prompt=non_empty_prompt,
        steps=valid_steps,
        cfg=valid_cfg,
        mu_shift=valid_mu_shift,
    )
    @settings(max_examples=100, deadline=None)
    def test_all_provided_uses_user_values_exactly(
        self, prompt: str, steps: int, cfg: float, mu_shift: float
    ):
        """When all sampler params are provided, user values are used exactly."""
        result = validate_params({
            "prompt": prompt,
            "steps": steps,
            "cfg": cfg,
            "mu_shift": mu_shift,
        })

        assert result.steps == steps
        assert result.cfg == cfg
        assert result.mu_shift == mu_shift


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 9: Seed determinism
# ---------------------------------------------------------------------------


class TestPropertySeedDeterminism:
    """Property 9: For any generation request where a seed is explicitly provided,
    that exact seed SHALL be used. For any request where no seed is provided,
    a random seed SHALL be generated, recorded, and used — such that the Job record
    alone is sufficient to reproduce the seed used.

    **Validates: Requirements 5.4**
    """

    @given(seed=valid_seed)
    @settings(max_examples=100, deadline=None)
    def test_explicit_seed_used_as_is(self, seed: int):
        """An explicit seed is returned unchanged by resolve_seed."""
        result = resolve_seed(seed)
        assert result == seed, (
            f"Explicit seed {seed} should be used as-is, got {result}"
        )

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_none_seed_generates_valid_random(self, data):
        """When no seed is provided (None), a random seed in [0, 2^32-1] is produced."""
        result = resolve_seed(None)

        # The result must be a valid integer in the seed range
        assert isinstance(result, int)
        assert SEED_MIN <= result <= SEED_MAX, (
            f"Generated seed {result} outside valid range [{SEED_MIN}, {SEED_MAX}]"
        )

    @given(seed=valid_seed)
    @settings(max_examples=100, deadline=None)
    def test_explicit_seed_is_deterministic(self, seed: int):
        """Calling resolve_seed with the same explicit seed always returns that seed."""
        result1 = resolve_seed(seed)
        result2 = resolve_seed(seed)
        assert result1 == result2 == seed


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 10: Parameter bounds validation
# ---------------------------------------------------------------------------


class TestPropertyParameterBounds:
    """Property 10: For any parameter value, the system SHALL accept it if and only
    if it falls within the defined bounds (steps: 1–100, seed: 0–2³²-1,
    width: 512–2048 multiples of 64, height: 512–2048 multiples of 64,
    batch_size: 1–16, batch_count: 1–100) and reject out-of-bounds values
    with a specific validation message.

    **Validates: Requirements 5.5**
    """

    @given(
        prompt=non_empty_prompt,
        steps=valid_steps,
        seed=valid_seed,
        width=valid_width,
        height=valid_height,
        batch_size=valid_batch_size,
        batch_count=valid_batch_count,
        precision=valid_precision,
    )
    @settings(max_examples=100, deadline=None)
    def test_in_bounds_params_accepted(
        self,
        prompt: str,
        steps: int,
        seed: int,
        width: int,
        height: int,
        batch_size: int,
        batch_count: int,
        precision: str,
    ):
        """All in-bounds parameters are accepted without error."""
        params = {
            "prompt": prompt,
            "steps": steps,
            "seed": seed,
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "batch_count": batch_count,
            "precision": precision,
        }
        result = validate_params(params)

        assert result.steps == steps
        assert result.seed == seed
        assert result.width == width
        assert result.height == height
        assert result.batch_size == batch_size
        assert result.batch_count == batch_count
        assert result.precision == precision

    @given(
        prompt=non_empty_prompt,
        bad_steps=st.one_of(
            st.integers(max_value=STEPS_MIN - 1),
            st.integers(min_value=STEPS_MAX + 1, max_value=10000),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_out_of_bounds_steps_rejected(self, prompt: str, bad_steps: int):
        """Steps outside [1, 100] are rejected with a validation message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "steps": bad_steps})
        msg = str(exc_info.value).lower()
        assert "steps" in msg

    @given(
        prompt=non_empty_prompt,
        bad_seed=st.one_of(
            st.integers(max_value=SEED_MIN - 1),
            st.integers(min_value=SEED_MAX + 1, max_value=SEED_MAX + 10000),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_out_of_bounds_seed_rejected(self, prompt: str, bad_seed: int):
        """Seed outside [0, 2^32-1] is rejected with a validation message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "seed": bad_seed})
        msg = str(exc_info.value).lower()
        assert "seed" in msg

    @given(
        prompt=non_empty_prompt,
        bad_width=st.one_of(
            st.integers(min_value=1, max_value=WIDTH_MIN - 1),
            st.integers(min_value=WIDTH_MAX + 1, max_value=5000),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_out_of_bounds_width_rejected(self, prompt: str, bad_width: int):
        """Width outside [512, 2048] is rejected with a validation message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "width": bad_width})
        msg = str(exc_info.value).lower()
        assert "width" in msg

    @given(
        prompt=non_empty_prompt,
        bad_height=st.one_of(
            st.integers(min_value=1, max_value=HEIGHT_MIN - 1),
            st.integers(min_value=HEIGHT_MAX + 1, max_value=5000),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_out_of_bounds_height_rejected(self, prompt: str, bad_height: int):
        """Height outside [512, 2048] is rejected with a validation message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "height": bad_height})
        msg = str(exc_info.value).lower()
        assert "height" in msg

    @given(
        prompt=non_empty_prompt,
        valid_base=st.integers(min_value=WIDTH_MIN // SIZE_MULTIPLE, max_value=WIDTH_MAX // SIZE_MULTIPLE),
    )
    @settings(max_examples=100, deadline=None)
    def test_width_not_multiple_of_64_rejected(self, prompt: str, valid_base: int):
        """Width that is not a multiple of 64 is rejected."""
        # Create a width that's in range but not a multiple of 64
        bad_width = valid_base * SIZE_MULTIPLE + 1  # Off by one from a valid multiple
        assume(WIDTH_MIN <= bad_width <= WIDTH_MAX)

        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "width": bad_width})
        msg = str(exc_info.value).lower()
        assert "width" in msg
        assert "multiple" in msg or "64" in msg

    @given(
        prompt=non_empty_prompt,
        valid_base=st.integers(min_value=HEIGHT_MIN // SIZE_MULTIPLE, max_value=HEIGHT_MAX // SIZE_MULTIPLE),
    )
    @settings(max_examples=100, deadline=None)
    def test_height_not_multiple_of_64_rejected(self, prompt: str, valid_base: int):
        """Height that is not a multiple of 64 is rejected."""
        bad_height = valid_base * SIZE_MULTIPLE + 1
        assume(HEIGHT_MIN <= bad_height <= HEIGHT_MAX)

        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "height": bad_height})
        msg = str(exc_info.value).lower()
        assert "height" in msg
        assert "multiple" in msg or "64" in msg

    @given(
        prompt=non_empty_prompt,
        bad_batch_size=st.one_of(
            st.integers(max_value=BATCH_SIZE_MIN - 1),
            st.integers(min_value=BATCH_SIZE_MAX + 1, max_value=100),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_out_of_bounds_batch_size_rejected(self, prompt: str, bad_batch_size: int):
        """Batch size outside [1, 16] is rejected with a validation message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "batch_size": bad_batch_size})
        msg = str(exc_info.value).lower()
        assert "batch size" in msg or "batch_size" in msg

    @given(
        prompt=non_empty_prompt,
        bad_batch_count=st.one_of(
            st.integers(max_value=BATCH_COUNT_MIN - 1),
            st.integers(min_value=BATCH_COUNT_MAX + 1, max_value=1000),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_out_of_bounds_batch_count_rejected(self, prompt: str, bad_batch_count: int):
        """Batch count outside [1, 100] is rejected with a validation message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": prompt, "batch_count": bad_batch_count})
        msg = str(exc_info.value).lower()
        assert "batch count" in msg or "batch_count" in msg


# ---------------------------------------------------------------------------
# Feature: cinderworks, Property 11: Batch produces correct image count
#          with correct per-image seeds
# ---------------------------------------------------------------------------


class TestPropertyBatchImageCount:
    """Property 11: For any (batch_size, batch_count, base_seed) tuple where
    batch_size ∈ [1,16] and batch_count ∈ [1,100], the system SHALL produce
    exactly batch_size × batch_count images, and image i within each batch
    SHALL use seed value (base_seed + i).

    **Validates: Requirements 6.1, 6.2**
    """

    @given(
        prompt=non_empty_prompt,
        batch_size=st.integers(min_value=1, max_value=4),
        batch_count=st.integers(min_value=1, max_value=4),
        base_seed=st.integers(min_value=0, max_value=2**32 - 100),
    )
    @settings(max_examples=100, deadline=None)
    def test_correct_total_image_count(
        self, prompt: str, batch_size: int, batch_count: int, base_seed: int
    ):
        """generate() produces exactly batch_size × batch_count images."""
        # Use a permissive VRAMManager (large total_vram)
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        params = {
            "prompt": prompt,
            "batch_size": batch_size,
            "batch_count": batch_count,
            "seed": base_seed,
            "_vram_manager": vram_mgr,
        }

        # Consume the generator to get the final result
        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None, "generate() should yield a final dict result"

        expected_total = batch_size * batch_count
        assert result["total_images"] == expected_total
        assert len(result["images"]) == expected_total
        assert len(result["seeds"]) == expected_total

    @given(
        prompt=non_empty_prompt,
        batch_size=st.integers(min_value=1, max_value=4),
        batch_count=st.integers(min_value=1, max_value=4),
        base_seed=st.integers(min_value=0, max_value=2**32 - 100),
    )
    @settings(max_examples=100, deadline=None)
    def test_correct_per_image_seeds(
        self, prompt: str, batch_size: int, batch_count: int, base_seed: int
    ):
        """Image i uses seed value (base_seed + i)."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        params = {
            "prompt": prompt,
            "batch_size": batch_size,
            "batch_count": batch_count,
            "seed": base_seed,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None

        # Each image i should have seed = base_seed + i
        expected_seeds = [base_seed + i for i in range(batch_size * batch_count)]
        assert result["seeds"] == expected_seeds, (
            f"Expected seeds {expected_seeds}, got {result['seeds']}"
        )

    @given(
        prompt=non_empty_prompt,
        batch_size=st.integers(min_value=1, max_value=4),
        batch_count=st.integers(min_value=1, max_value=4),
        base_seed=st.integers(min_value=0, max_value=2**32 - 100),
    )
    @settings(max_examples=100, deadline=None)
    def test_base_seed_recorded_in_result(
        self, prompt: str, batch_size: int, batch_count: int, base_seed: int
    ):
        """The base_seed is correctly recorded in the final result."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        params = {
            "prompt": prompt,
            "batch_size": batch_size,
            "batch_count": batch_count,
            "seed": base_seed,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert result["base_seed"] == base_seed
        assert result["params"]["seed"] == base_seed


# ===========================================================================
# UNIT TESTS — Task 7.3: Write unit tests for Krea 2 backend
# ===========================================================================


class TestEncodeOffloadSampleDecodeOrder:
    """Test encode→offload→sample→decode order via mocked harness.

    Verifies that the generation pipeline follows the correct sequence:
    1. acquire(text_encoder) — load encoder
    2. encode prompt
    3. release(text_encoder) — offload encoder
    4. acquire(dit) — load diffusion model
    5. sample
    6. release(dit) — offload diffusion model
    7. decode

    **Validates: Requirements 5.1**
    """

    def test_pipeline_order_via_vram_manager(self):
        """Verify acquire/release calls follow encode→offload→sample→decode order."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        # Track acquire/release calls
        call_log: list[str] = []
        original_acquire = vram_mgr.acquire
        original_release = vram_mgr.release

        def tracking_acquire(tenant):
            call_log.append(f"acquire:{tenant.name}")
            original_acquire(tenant)

        def tracking_release(tenant):
            call_log.append(f"release:{tenant.name}")
            original_release(tenant)

        vram_mgr.acquire = tracking_acquire
        vram_mgr.release = tracking_release

        params = {
            "prompt": "a beautiful sunset over mountains",
            "seed": 42,
            "_vram_manager": vram_mgr,
        }

        # Exhaust the generator
        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None, "generate() must yield a final dict result"

        # Verify the ordering: text_encoder acquire/release before dit acquire/release
        assert "acquire:text_encoder" in call_log
        assert "release:text_encoder" in call_log
        assert "acquire:dit" in call_log
        assert "release:dit" in call_log

        enc_acquire_idx = call_log.index("acquire:text_encoder")
        enc_release_idx = call_log.index("release:text_encoder")
        dit_acquire_idx = call_log.index("acquire:dit")
        dit_release_idx = call_log.index("release:dit")

        # Encode sequence: acquire encoder → release encoder (offload)
        assert enc_acquire_idx < enc_release_idx, (
            "Text encoder must be acquired before released"
        )
        # Offload before sampling: encoder released before dit acquired
        assert enc_release_idx < dit_acquire_idx, (
            "Text encoder must be offloaded before DiT is acquired"
        )
        # Sampling sequence: acquire dit → release dit
        assert dit_acquire_idx < dit_release_idx, (
            "DiT must be acquired before released"
        )

    def test_encoder_not_resident_during_dit_sampling(self):
        """Verify that text encoder is NOT resident when DiT is loaded."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        # Track which tenants are resident at each acquire call
        tenants_resident_at_dit_acquire: list[str | None] = []
        original_acquire = vram_mgr.acquire

        def tracking_acquire(tenant):
            if tenant.name == "dit":
                resident = vram_mgr.resident
                tenants_resident_at_dit_acquire.append(
                    resident.name if resident else None
                )
            original_acquire(tenant)

        vram_mgr.acquire = tracking_acquire

        params = {
            "prompt": "test prompt for residency check",
            "seed": 100,
            "_vram_manager": vram_mgr,
        }

        # Exhaust generator
        for _ in generate(params):
            pass

        # At the point DiT is acquired, text_encoder should NOT be resident
        assert len(tenants_resident_at_dit_acquire) > 0
        assert tenants_resident_at_dit_acquire[0] is None, (
            "Text encoder should not be resident when DiT is acquired"
        )


class TestTurboDefaultsApplied:
    """Test Turbo defaults applied when params omitted.

    **Validates: Requirements 5.2**
    """

    def test_defaults_applied_with_only_prompt(self):
        """validate_params with only a prompt applies Turbo defaults."""
        result = validate_params({"prompt": "test"})

        assert result.steps == 8, f"Expected steps=8, got {result.steps}"
        assert result.cfg == 1.0, f"Expected cfg=1.0, got {result.cfg}"
        assert result.mu_shift == 1.15, f"Expected mu_shift=1.15, got {result.mu_shift}"

    def test_defaults_include_resolution_and_batch(self):
        """validate_params with only a prompt also sets correct defaults for other params."""
        result = validate_params({"prompt": "test"})

        assert result.width == 1024
        assert result.height == 1024
        assert result.precision == "bf16"
        assert result.batch_size == 1
        assert result.batch_count == 1
        assert result.seed is None

    def test_explicit_overrides_preserved(self):
        """Explicitly provided values override Turbo defaults."""
        result = validate_params({
            "prompt": "test",
            "steps": 20,
            "cfg": 7.5,
            "mu_shift": 2.0,
        })

        assert result.steps == 20
        assert result.cfg == 7.5
        assert result.mu_shift == 2.0


class TestEmptyPromptRefused:
    """Test empty prompt refused with plain-language message.

    **Validates: Requirements 5.7**
    """

    def test_empty_string_rejected(self):
        """An empty string prompt raises ValueError with a plain-language message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": ""})
        msg = str(exc_info.value)
        # Must be plain language, not a traceback
        assert "prompt" in msg.lower()
        assert "required" in msg.lower()

    def test_whitespace_only_rejected(self):
        """A whitespace-only prompt raises ValueError with a plain-language message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": "   "})
        msg = str(exc_info.value)
        assert "prompt" in msg.lower()
        assert "required" in msg.lower()

    def test_missing_prompt_key_rejected(self):
        """A missing prompt key raises ValueError with a plain-language message."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({})
        msg = str(exc_info.value)
        assert "prompt" in msg.lower()

    def test_error_message_is_user_friendly(self):
        """The error message is a complete sentence suitable for UI display."""
        with pytest.raises(ValueError) as exc_info:
            validate_params({"prompt": ""})
        msg = str(exc_info.value)
        # Should not contain Python jargon
        assert "TypeError" not in msg
        assert "NoneType" not in msg
        assert "traceback" not in msg.lower()
        # Should be a readable sentence
        assert len(msg) > 10, "Error message should be a complete sentence"


class TestBatchSizeProducesCorrectImages:
    """Test batch_size > 1 produces correct number of images with sequential seeds.

    **Validates: Requirements 6.1**
    """

    def test_batch_size_3_produces_3_images(self):
        """batch_size=3 with seed=42 produces 3 images with seeds 42, 43, 44."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        params = {
            "prompt": "test",
            "batch_size": 3,
            "seed": 42,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None, "generate() must yield a final dict result"
        assert len(result["images"]) == 3, (
            f"Expected 3 images, got {len(result['images'])}"
        )
        assert result["seeds"] == [42, 43, 44], (
            f"Expected seeds [42, 43, 44], got {result['seeds']}"
        )

    def test_batch_size_1_produces_1_image(self):
        """batch_size=1 (default) produces exactly 1 image with the base seed."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        params = {
            "prompt": "single image test",
            "batch_size": 1,
            "seed": 100,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert len(result["images"]) == 1
        assert result["seeds"] == [100]

    def test_batch_size_with_batch_count(self):
        """batch_size=2, batch_count=2, seed=10 produces 4 images with seeds 10–13."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        params = {
            "prompt": "multi-batch test",
            "batch_size": 2,
            "batch_count": 2,
            "seed": 10,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        assert len(result["images"]) == 4, (
            f"Expected 4 images (2×2), got {len(result['images'])}"
        )
        assert result["seeds"] == [10, 11, 12, 13], (
            f"Expected seeds [10, 11, 12, 13], got {result['seeds']}"
        )
        assert result["total_images"] == 4

    def test_seed_continuity_across_batches(self):
        """Seeds are sequential across batch boundaries, not reset per batch."""
        vram_mgr = VRAMManager(total_vram=100_000_000_000)

        params = {
            "prompt": "seed continuity test",
            "batch_size": 3,
            "batch_count": 2,
            "seed": 0,
            "_vram_manager": vram_mgr,
        }

        result = None
        for output in generate(params):
            if isinstance(output, dict):
                result = output

        assert result is not None
        # 6 images total, seeds 0 through 5
        assert result["seeds"] == [0, 1, 2, 3, 4, 5], (
            f"Expected seeds [0,1,2,3,4,5], got {result['seeds']}"
        )
