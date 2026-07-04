"""Tests for core/lora_manager.py — LoRA discovery, validation, dataclasses, and pipeline application."""

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from studio.core.lora_manager import (
    LoRAEntry,
    LoRAStack,
    _LORA_VRAM_ESTIMATE_BYTES,
    add_lora_to_stack,
    apply_loras,
    estimate_lora_stack_vram,
    get_loras_dir,
    remove_lora_from_stack,
    scan_loras,
    unload_loras,
    validate_lora_file,
)
from studio.core.vram_manager import Tenant, VRAMManager
from studio.config import Config


# ---------------------------------------------------------------------------
# Helpers — create fake .safetensors files
# ---------------------------------------------------------------------------


def _make_valid_safetensors(path: Path, metadata: dict | None = None) -> None:
    """Write a minimal valid .safetensors file (header only, no tensor data)."""
    if metadata is None:
        metadata = {"__metadata__": {"format": "pt"}}
    header_json = json.dumps(metadata).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    path.write_bytes(header_len + header_json)


def _make_invalid_safetensors(path: Path) -> None:
    """Write a .safetensors file with garbage content."""
    path.write_bytes(b"\x00" * 16)  # Zero header length + garbage


def _make_truncated_safetensors(path: Path) -> None:
    """Write a .safetensors file that claims a longer header than available."""
    header_len = struct.pack("<Q", 1000)  # Claims 1000 bytes
    path.write_bytes(header_len + b"short")  # Only 5 bytes of content


# ---------------------------------------------------------------------------
# LoRAEntry dataclass tests
# ---------------------------------------------------------------------------


class TestLoRAEntry:
    def test_default_weight(self):
        """LoRAEntry defaults weight to 1.0."""
        entry = LoRAEntry(file_path=Path("/some/lora.safetensors"), filename="lora")
        assert entry.weight == 1.0

    def test_custom_weight(self):
        """LoRAEntry accepts custom weight."""
        entry = LoRAEntry(
            file_path=Path("/some/lora.safetensors"), filename="lora", weight=0.75
        )
        assert entry.weight == 0.75

    def test_fields(self):
        """LoRAEntry stores file_path and filename correctly."""
        p = Path("/studio/loras/style_anime.safetensors")
        entry = LoRAEntry(file_path=p, filename="style_anime", weight=1.2)
        assert entry.file_path == p
        assert entry.filename == "style_anime"
        assert entry.weight == 1.2


# ---------------------------------------------------------------------------
# LoRAStack dataclass tests
# ---------------------------------------------------------------------------


class TestLoRAStack:
    def test_default_empty(self):
        """LoRAStack starts with an empty entries list by default."""
        stack = LoRAStack()
        assert stack.entries == []

    def test_with_entries(self):
        """LoRAStack holds a list of LoRAEntry objects."""
        entries = [
            LoRAEntry(file_path=Path("/a.safetensors"), filename="a"),
            LoRAEntry(file_path=Path("/b.safetensors"), filename="b", weight=0.5),
        ]
        stack = LoRAStack(entries=entries)
        assert len(stack.entries) == 2
        assert stack.entries[0].filename == "a"
        assert stack.entries[1].weight == 0.5

    def test_independent_instances(self):
        """Each LoRAStack instance has its own entries list."""
        stack1 = LoRAStack()
        stack2 = LoRAStack()
        stack1.entries.append(
            LoRAEntry(file_path=Path("/x.safetensors"), filename="x")
        )
        assert len(stack2.entries) == 0


# ---------------------------------------------------------------------------
# validate_lora_file tests
# ---------------------------------------------------------------------------


class TestValidateLoraFile:
    def test_valid_file(self, tmp_path):
        """Returns True for a properly formatted .safetensors file."""
        f = tmp_path / "valid.safetensors"
        _make_valid_safetensors(f)
        assert validate_lora_file(f) is True

    def test_invalid_garbage(self, tmp_path):
        """Returns False for garbage data (zero header length)."""
        f = tmp_path / "garbage.safetensors"
        _make_invalid_safetensors(f)
        assert validate_lora_file(f) is False

    def test_truncated_file(self, tmp_path):
        """Returns False when header is truncated."""
        f = tmp_path / "truncated.safetensors"
        _make_truncated_safetensors(f)
        assert validate_lora_file(f) is False

    def test_too_small(self, tmp_path):
        """Returns False when file is smaller than 8 bytes."""
        f = tmp_path / "tiny.safetensors"
        f.write_bytes(b"\x01\x02\x03")
        assert validate_lora_file(f) is False

    def test_nonexistent_file(self, tmp_path):
        """Returns False for a file that doesn't exist."""
        f = tmp_path / "nonexistent.safetensors"
        assert validate_lora_file(f) is False

    def test_invalid_json_header(self, tmp_path):
        """Returns False when header bytes are not valid JSON."""
        f = tmp_path / "badjson.safetensors"
        bad_header = b"not valid json at all!!"
        header_len = struct.pack("<Q", len(bad_header))
        f.write_bytes(header_len + bad_header)
        assert validate_lora_file(f) is False

    def test_valid_complex_metadata(self, tmp_path):
        """Returns True for a file with complex but valid JSON header."""
        f = tmp_path / "complex.safetensors"
        metadata = {
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 768],
                "data_offsets": [0, 6144],
            },
            "__metadata__": {"format": "pt", "ss_network_dim": "32"},
        }
        _make_valid_safetensors(f, metadata)
        assert validate_lora_file(f) is True


# ---------------------------------------------------------------------------
# scan_loras tests
# ---------------------------------------------------------------------------


class TestScanLoras:
    def test_empty_directory(self, tmp_path):
        """Returns empty list for an empty directory."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()
        result = scan_loras(loras_dir)
        assert result == []

    def test_creates_missing_directory(self, tmp_path):
        """Creates the directory if it doesn't exist (Requirement 1.3)."""
        loras_dir = tmp_path / "new_loras"
        assert not loras_dir.exists()
        result = scan_loras(loras_dir)
        assert loras_dir.exists()
        assert result == []

    def test_finds_valid_files(self, tmp_path):
        """Returns stems of valid .safetensors files."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        _make_valid_safetensors(loras_dir / "style_anime.safetensors")
        _make_valid_safetensors(loras_dir / "subject_character.safetensors")

        result = scan_loras(loras_dir)
        assert result == ["style_anime", "subject_character"]

    def test_skips_invalid_files(self, tmp_path):
        """Skips files that fail header validation (Requirement 1.4)."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        _make_valid_safetensors(loras_dir / "good.safetensors")
        _make_invalid_safetensors(loras_dir / "bad.safetensors")

        result = scan_loras(loras_dir)
        assert result == ["good"]

    def test_ignores_non_safetensors_files(self, tmp_path):
        """Only considers files with .safetensors extension (Requirement 1.1)."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        _make_valid_safetensors(loras_dir / "valid.safetensors")
        (loras_dir / "readme.txt").write_text("not a lora")
        (loras_dir / "model.ckpt").write_bytes(b"\x00" * 100)

        result = scan_loras(loras_dir)
        assert result == ["valid"]

    def test_sorted_output(self, tmp_path):
        """Results are returned in sorted order (deterministic)."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        _make_valid_safetensors(loras_dir / "zebra.safetensors")
        _make_valid_safetensors(loras_dir / "alpha.safetensors")
        _make_valid_safetensors(loras_dir / "middle.safetensors")

        result = scan_loras(loras_dir)
        assert result == ["alpha", "middle", "zebra"]

    def test_permissions_fallback(self, tmp_path, monkeypatch):
        """Returns empty list if directory creation fails due to permissions."""
        # Use a path that we'll make the mkdir fail for
        bad_dir = tmp_path / "nope" / "loras"

        # Monkeypatch Path.mkdir to raise PermissionError
        original_mkdir = Path.mkdir

        def failing_mkdir(self, *args, **kwargs):
            if str(self) == str(bad_dir):
                raise PermissionError("No permission")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)

        result = scan_loras(bad_dir)
        assert result == []


# ---------------------------------------------------------------------------
# get_loras_dir tests
# ---------------------------------------------------------------------------


class TestGetLorasDir:
    def test_default_path(self):
        """Default loras dir is studio/loras/."""
        result = get_loras_dir()
        assert result == Config.LORAS_DIR

    def test_returns_pathlib_path(self):
        """Returns a pathlib.Path instance."""
        result = get_loras_dir()
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# add_lora_to_stack tests
# ---------------------------------------------------------------------------


class TestAddLoraToStack:
    def test_add_to_empty_stack(self, tmp_path):
        """Adding a LoRA to an empty stack returns a stack with one entry."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        result_json, message = add_lora_to_stack(
            "style_anime", 1.0, "[]", loras_dir=loras_dir
        )
        result = json.loads(result_json)

        assert len(result) == 1
        assert result[0]["path"] == str(loras_dir / "style_anime.safetensors")
        assert result[0]["weight"] == 1.0
        assert message == ""

    def test_add_with_custom_weight(self, tmp_path):
        """Adding a LoRA records the specified weight."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        result_json, message = add_lora_to_stack(
            "style_anime", 0.8, "[]", loras_dir=loras_dir
        )
        result = json.loads(result_json)

        assert result[0]["weight"] == 0.8
        assert message == ""

    def test_add_multiple_loras(self, tmp_path):
        """Adding multiple distinct LoRAs grows the stack."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        stack_json, _ = add_lora_to_stack(
            "style_anime", 1.0, "[]", loras_dir=loras_dir
        )
        stack_json, _ = add_lora_to_stack(
            "subject_character", 0.5, stack_json, loras_dir=loras_dir
        )
        result = json.loads(stack_json)

        assert len(result) == 2
        assert result[0]["path"] == str(loras_dir / "style_anime.safetensors")
        assert result[1]["path"] == str(loras_dir / "subject_character.safetensors")

    def test_reject_duplicate(self, tmp_path):
        """Adding the same LoRA file path twice is refused with a message."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        stack_json, _ = add_lora_to_stack(
            "style_anime", 1.0, "[]", loras_dir=loras_dir
        )
        result_json, message = add_lora_to_stack(
            "style_anime", 0.5, stack_json, loras_dir=loras_dir
        )
        result = json.loads(result_json)

        # Stack unchanged — still one entry
        assert len(result) == 1
        assert result[0]["weight"] == 1.0  # Original weight preserved
        assert "'style_anime' is already in the LoRA stack." == message

    def test_add_to_null_json(self, tmp_path):
        """Handles 'null' JSON gracefully as empty stack."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        result_json, message = add_lora_to_stack(
            "my_lora", 1.0, "null", loras_dir=loras_dir
        )
        result = json.loads(result_json)

        assert len(result) == 1
        assert message == ""

    def test_add_to_empty_string(self, tmp_path):
        """Handles empty string JSON gracefully as empty stack."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        result_json, message = add_lora_to_stack(
            "my_lora", 1.0, "", loras_dir=loras_dir
        )
        result = json.loads(result_json)

        assert len(result) == 1
        assert message == ""


# ---------------------------------------------------------------------------
# remove_lora_from_stack tests
# ---------------------------------------------------------------------------


class TestRemoveLoraFromStack:
    def test_remove_existing(self, tmp_path):
        """Removing an existing LoRA reduces the stack."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        # Build a stack with two entries
        stack_json, _ = add_lora_to_stack(
            "style_anime", 1.0, "[]", loras_dir=loras_dir
        )
        stack_json, _ = add_lora_to_stack(
            "subject_character", 0.5, stack_json, loras_dir=loras_dir
        )

        # Remove first entry
        result_json, message = remove_lora_from_stack(
            "style_anime", stack_json, loras_dir=loras_dir
        )
        result = json.loads(result_json)

        assert len(result) == 1
        assert result[0]["path"] == str(loras_dir / "subject_character.safetensors")
        assert message == ""

    def test_remove_nonexistent(self, tmp_path):
        """Removing a LoRA not in the stack returns an informational message."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        stack_json, _ = add_lora_to_stack(
            "style_anime", 1.0, "[]", loras_dir=loras_dir
        )

        result_json, message = remove_lora_from_stack(
            "nonexistent_lora", stack_json, loras_dir=loras_dir
        )
        result = json.loads(result_json)

        # Stack unchanged
        assert len(result) == 1
        assert "'nonexistent_lora' was not found in the LoRA stack." == message

    def test_remove_from_empty_stack(self, tmp_path):
        """Removing from an empty stack returns informational message."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        result_json, message = remove_lora_from_stack(
            "any_lora", "[]", loras_dir=loras_dir
        )
        result = json.loads(result_json)

        assert result == []
        assert "'any_lora' was not found in the LoRA stack." == message

    def test_remove_last_entry(self, tmp_path):
        """Removing the last entry results in an empty stack."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        stack_json, _ = add_lora_to_stack(
            "only_lora", 1.0, "[]", loras_dir=loras_dir
        )

        result_json, message = remove_lora_from_stack(
            "only_lora", stack_json, loras_dir=loras_dir
        )
        result = json.loads(result_json)

        assert result == []
        assert message == ""

    def test_remove_preserves_order(self, tmp_path):
        """Removing a middle entry preserves order of remaining entries."""
        loras_dir = tmp_path / "loras"
        loras_dir.mkdir()

        stack_json, _ = add_lora_to_stack("a", 1.0, "[]", loras_dir=loras_dir)
        stack_json, _ = add_lora_to_stack("b", 0.5, stack_json, loras_dir=loras_dir)
        stack_json, _ = add_lora_to_stack("c", 0.8, stack_json, loras_dir=loras_dir)

        result_json, _ = remove_lora_from_stack("b", stack_json, loras_dir=loras_dir)
        result = json.loads(result_json)

        assert len(result) == 2
        assert result[0]["path"] == str(loras_dir / "a.safetensors")
        assert result[1]["path"] == str(loras_dir / "c.safetensors")


# ---------------------------------------------------------------------------
# estimate_lora_stack_vram tests
# ---------------------------------------------------------------------------


class TestEstimateLoraStackVram:
    def test_empty_stack(self):
        """Empty stack has zero VRAM estimate."""
        stack = LoRAStack()
        assert estimate_lora_stack_vram(stack) == 0

    def test_single_lora(self):
        """Single LoRA returns one unit of the estimate constant."""
        stack = LoRAStack(
            entries=[LoRAEntry(file_path=Path("/a.safetensors"), filename="a")]
        )
        assert estimate_lora_stack_vram(stack) == _LORA_VRAM_ESTIMATE_BYTES

    def test_multiple_loras(self):
        """N LoRAs return N * estimate constant."""
        entries = [
            LoRAEntry(file_path=Path(f"/{i}.safetensors"), filename=str(i))
            for i in range(5)
        ]
        stack = LoRAStack(entries=entries)
        assert estimate_lora_stack_vram(stack) == 5 * _LORA_VRAM_ESTIMATE_BYTES


# ---------------------------------------------------------------------------
# apply_loras tests
# ---------------------------------------------------------------------------


class TestApplyLoras:
    def _make_pipeline_mock(self):
        """Create a mock pipeline with load_lora_weights and set_adapters."""
        pipeline = MagicMock()
        pipeline.load_lora_weights = MagicMock()
        pipeline.set_adapters = MagicMock()
        pipeline.unload_lora_weights = MagicMock()
        return pipeline

    def _make_vram_manager(self, total_vram: int = 24_000_000_000) -> VRAMManager:
        """Create a VRAMManager with specified capacity."""
        return VRAMManager(total_vram=total_vram)

    def test_empty_stack_no_op(self):
        """Empty LoRA stack does nothing to the pipeline."""
        pipeline = self._make_pipeline_mock()
        vram_mgr = self._make_vram_manager()
        stack = LoRAStack()

        apply_loras(pipeline, stack, vram_manager=vram_mgr)

        pipeline.load_lora_weights.assert_not_called()
        pipeline.set_adapters.assert_not_called()

    def test_single_lora_applied(self):
        """Single LoRA is loaded and adapters are set with correct weight."""
        pipeline = self._make_pipeline_mock()
        vram_mgr = self._make_vram_manager()
        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/style_anime.safetensors"),
                    filename="style_anime",
                    weight=0.8,
                )
            ]
        )

        apply_loras(pipeline, stack, vram_manager=vram_mgr)

        pipeline.load_lora_weights.assert_called_once_with(
            str(Path("/loras/style_anime.safetensors")),
            adapter_name="lora_0_style_anime",
        )
        pipeline.set_adapters.assert_called_once_with(
            ["lora_0_style_anime"],
            adapter_weights=[0.8],
        )

    def test_multiple_loras_applied_in_order(self):
        """Multiple LoRAs are applied in stack order with correct weights."""
        pipeline = self._make_pipeline_mock()
        vram_mgr = self._make_vram_manager()
        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/style.safetensors"),
                    filename="style",
                    weight=0.8,
                ),
                LoRAEntry(
                    file_path=Path("/loras/subject.safetensors"),
                    filename="subject",
                    weight=1.0,
                ),
                LoRAEntry(
                    file_path=Path("/loras/detail.safetensors"),
                    filename="detail",
                    weight=0.5,
                ),
            ]
        )

        apply_loras(pipeline, stack, vram_manager=vram_mgr)

        # Verify load order
        assert pipeline.load_lora_weights.call_count == 3
        calls = pipeline.load_lora_weights.call_args_list
        assert calls[0] == call(
            str(Path("/loras/style.safetensors")),
            adapter_name="lora_0_style",
        )
        assert calls[1] == call(
            str(Path("/loras/subject.safetensors")),
            adapter_name="lora_1_subject",
        )
        assert calls[2] == call(
            str(Path("/loras/detail.safetensors")),
            adapter_name="lora_2_detail",
        )

        # Verify set_adapters called with all names and weights in order
        pipeline.set_adapters.assert_called_once_with(
            ["lora_0_style", "lora_1_subject", "lora_2_detail"],
            adapter_weights=[0.8, 1.0, 0.5],
        )

    def test_vram_overflow_raises_before_loading(self):
        """VRAM overflow raises RuntimeError before any LoRA loading."""
        pipeline = self._make_pipeline_mock()
        # Give very small VRAM budget
        vram_mgr = self._make_vram_manager(total_vram=100_000_000)  # 100 MB

        # Add a resident tenant that takes up most of the budget
        tenant = Tenant(
            name="dit",
            estimated_bytes=50_000_000,
            load_fn=lambda: None,
            unload_fn=lambda: None,
        )
        vram_mgr.acquire(tenant)

        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/big.safetensors"),
                    filename="big",
                    weight=1.0,
                )
            ]
        )

        with pytest.raises(RuntimeError, match="Not enough VRAM"):
            apply_loras(pipeline, stack, vram_manager=vram_mgr)

        # Pipeline should NOT have been touched
        pipeline.load_lora_weights.assert_not_called()

    def test_failed_lora_identified_in_error(self):
        """Failed LoRA is identified by filename in the error message."""
        pipeline = self._make_pipeline_mock()
        vram_mgr = self._make_vram_manager()

        # Second LoRA fails to load
        pipeline.load_lora_weights.side_effect = [
            None,  # First succeeds
            OSError("File corrupted"),  # Second fails
        ]

        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/good.safetensors"),
                    filename="good",
                    weight=1.0,
                ),
                LoRAEntry(
                    file_path=Path("/loras/corrupted.safetensors"),
                    filename="corrupted",
                    weight=0.8,
                ),
            ]
        )

        with pytest.raises(RuntimeError, match="corrupted"):
            apply_loras(pipeline, stack, vram_manager=vram_mgr)

    def test_failed_lora_triggers_cleanup(self):
        """When a LoRA fails, unload_lora_weights is called for cleanup."""
        pipeline = self._make_pipeline_mock()
        vram_mgr = self._make_vram_manager()

        pipeline.load_lora_weights.side_effect = OSError("File not found")

        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/missing.safetensors"),
                    filename="missing",
                    weight=1.0,
                )
            ]
        )

        with pytest.raises(RuntimeError):
            apply_loras(pipeline, stack, vram_manager=vram_mgr)

        # Cleanup was attempted
        pipeline.unload_lora_weights.assert_called_once()

    def test_vram_check_with_no_resident(self):
        """VRAM check works when no tenant is currently resident."""
        pipeline = self._make_pipeline_mock()
        # Budget that can fit LoRAs without a base model
        vram_mgr = self._make_vram_manager(total_vram=2_000_000_000)

        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/style.safetensors"),
                    filename="style",
                    weight=1.0,
                )
            ]
        )

        # Should not raise — 300 MB LoRA fits in 2 GB budget
        apply_loras(pipeline, stack, vram_manager=vram_mgr)
        pipeline.load_lora_weights.assert_called_once()

    def test_vram_error_message_includes_lora_count(self):
        """VRAM error message includes the number of LoRAs."""
        pipeline = self._make_pipeline_mock()
        # Tiny budget
        vram_mgr = self._make_vram_manager(total_vram=100_000_000)

        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path(f"/loras/lora{i}.safetensors"),
                    filename=f"lora{i}",
                    weight=1.0,
                )
                for i in range(3)
            ]
        )

        with pytest.raises(RuntimeError, match="3 LoRAs"):
            apply_loras(pipeline, stack, vram_manager=vram_mgr)

    def test_set_adapters_failure_triggers_cleanup(self):
        """If set_adapters fails, LoRAs are unloaded and RuntimeError raised."""
        pipeline = self._make_pipeline_mock()
        vram_mgr = self._make_vram_manager()

        pipeline.set_adapters.side_effect = RuntimeError("Incompatible adapters")

        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/style.safetensors"),
                    filename="style",
                    weight=1.0,
                )
            ]
        )

        with pytest.raises(RuntimeError, match="Could not apply LoRA weights"):
            apply_loras(pipeline, stack, vram_manager=vram_mgr)

        pipeline.unload_lora_weights.assert_called_once()


# ---------------------------------------------------------------------------
# unload_loras tests
# ---------------------------------------------------------------------------


class TestUnloadLoras:
    def test_unload_calls_pipeline(self):
        """unload_loras calls pipeline.unload_lora_weights()."""
        pipeline = MagicMock()
        pipeline.unload_lora_weights = MagicMock()

        unload_loras(pipeline)

        pipeline.unload_lora_weights.assert_called_once()

    def test_unload_does_not_crash_on_error(self):
        """unload_loras logs warning but does not raise on pipeline error."""
        pipeline = MagicMock()
        pipeline.unload_lora_weights.side_effect = RuntimeError("No adapters loaded")

        # Should not raise
        unload_loras(pipeline)

    def test_unload_restores_base_model(self):
        """After apply + unload cycle, base model is restored (no LoRAs active)."""
        pipeline = MagicMock()
        pipeline.load_lora_weights = MagicMock()
        pipeline.set_adapters = MagicMock()
        pipeline.unload_lora_weights = MagicMock()

        vram_mgr = VRAMManager(total_vram=24_000_000_000)
        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/style.safetensors"),
                    filename="style",
                    weight=1.0,
                )
            ]
        )

        # Apply then unload (lifecycle pattern)
        apply_loras(pipeline, stack, vram_manager=vram_mgr)
        unload_loras(pipeline)

        # Verify full lifecycle: load → set_adapters → unload
        pipeline.load_lora_weights.assert_called_once()
        pipeline.set_adapters.assert_called_once()
        pipeline.unload_lora_weights.assert_called_once()


# ---------------------------------------------------------------------------
# Forge-Neo-style loading behavior tests (key conversion, adapter names,
# failure messages)
# ---------------------------------------------------------------------------


from studio.core.lora_manager import (  # noqa: E402
    _convert_lora_state_dict,
    _lora_failure_message,
    _normalize_module_path,
    _sanitize_adapter_name,
    _split_lora_key,
)


class _FakeTensor:
    """Minimal stand-in for a torch tensor (no torch in the test env)."""

    def __init__(self, shape=(8, 4), value=1.0):
        self.shape = shape
        self.ndim = len(shape)
        self.value = value

    def __mul__(self, scalar):
        return _FakeTensor(self.shape, self.value * scalar)

    def __float__(self):
        return float(self.value)


class TestSanitizeAdapterName:
    def test_plain_name_unchanged(self):
        assert _sanitize_adapter_name("lora_0_style_anime") == "lora_0_style_anime"

    def test_dots_replaced(self):
        """Dots would crash torch ModuleDict — must be replaced."""
        assert _sanitize_adapter_name("lora_0_detail-v1.5") == "lora_0_detail_v1_5"

    def test_applied_during_load(self):
        """apply_loras passes a sanitized adapter name to the pipeline."""
        pipeline = MagicMock()
        vram_mgr = VRAMManager(total_vram=24_000_000_000)
        stack = LoRAStack(
            entries=[
                LoRAEntry(
                    file_path=Path("/loras/style.v1.0.safetensors"),
                    filename="style.v1.0",
                    weight=1.0,
                )
            ]
        )

        apply_loras(pipeline, stack, vram_manager=vram_mgr)

        _, kwargs = pipeline.load_lora_weights.call_args
        assert kwargs["adapter_name"] == "lora_0_style_v1_0"
        assert "." not in kwargs["adapter_name"]


class TestSplitLoraKey:
    def test_diffusers_peft_suffixes(self):
        assert _split_lora_key("transformer.blocks.0.to_q.lora_A.weight") == (
            "transformer.blocks.0.to_q",
            "A",
        )
        assert _split_lora_key("transformer.blocks.0.to_q.lora_B.weight") == (
            "transformer.blocks.0.to_q",
            "B",
        )

    def test_kohya_suffixes(self):
        assert _split_lora_key("lora_unet_blocks_0_to_q.lora_down.weight") == (
            "lora_unet_blocks_0_to_q",
            "A",
        )
        assert _split_lora_key("lora_unet_blocks_0_to_q.lora_up.weight") == (
            "lora_unet_blocks_0_to_q",
            "B",
        )
        assert _split_lora_key("lora_unet_blocks_0_to_q.alpha") == (
            "lora_unet_blocks_0_to_q",
            "alpha",
        )

    def test_unrecognized_returns_none(self):
        assert _split_lora_key("some.random.weight") is None


class TestNormalizeModulePath:
    def test_diffusers_passthrough(self):
        assert (
            _normalize_module_path("transformer.blocks.0.attn.to_q", {})
            == "transformer.blocks.0.attn.to_q"
        )

    def test_comfyui_prefix_remapped(self):
        assert (
            _normalize_module_path("diffusion_model.blocks.0.attn.to_q", {})
            == "transformer.blocks.0.attn.to_q"
        )

    def test_kohya_flattened_resolved_via_model_map(self):
        kohya_map = {"blocks_0_attn_to_q": "blocks.0.attn.to_q"}
        assert (
            _normalize_module_path("lora_unet_blocks_0_attn_to_q", kohya_map)
            == "transformer.blocks.0.attn.to_q"
        )

    def test_kohya_flattened_unresolvable_returns_none(self):
        assert _normalize_module_path("lora_unet_blocks_0_attn_to_q", {}) is None

    def test_bare_module_path_gets_transformer_prefix(self):
        assert (
            _normalize_module_path("blocks.0.attn.to_q", {})
            == "transformer.blocks.0.attn.to_q"
        )


class TestConvertLoraStateDict:
    def test_comfyui_format_converted(self):
        sd = {
            "diffusion_model.blocks.0.to_q.lora_down.weight": _FakeTensor((4, 16)),
            "diffusion_model.blocks.0.to_q.lora_up.weight": _FakeTensor((16, 4)),
        }
        converted, unmatched, te_skipped = _convert_lora_state_dict(sd, {})
        assert set(converted) == {
            "transformer.blocks.0.to_q.lora_A.weight",
            "transformer.blocks.0.to_q.lora_B.weight",
        }
        assert unmatched == []
        assert te_skipped == 0

    def test_alpha_folded_into_lora_b(self):
        """alpha/rank scaling is folded into the up (lora_B) tensor."""
        sd = {
            "diffusion_model.blocks.0.to_q.lora_down.weight": _FakeTensor((4, 16)),
            "diffusion_model.blocks.0.to_q.lora_up.weight": _FakeTensor((16, 4), value=2.0),
            "diffusion_model.blocks.0.to_q.alpha": _FakeTensor((), value=2.0),
        }
        converted, _, _ = _convert_lora_state_dict(sd, {})
        lora_b = converted["transformer.blocks.0.to_q.lora_B.weight"]
        # alpha=2.0, rank=4 -> scale 0.5; up value 2.0 * 0.5 = 1.0
        assert lora_b.value == pytest.approx(1.0)
        # No alpha keys leak into the converted dict
        assert not any(k.endswith(".alpha") for k in converted)

    def test_text_encoder_keys_skipped_not_unmatched(self):
        sd = {
            "lora_te_encoder_layers_0_q.lora_down.weight": _FakeTensor(),
            "diffusion_model.blocks.0.to_q.lora_A.weight": _FakeTensor(),
        }
        converted, unmatched, te_skipped = _convert_lora_state_dict(sd, {})
        assert te_skipped == 1
        assert unmatched == []
        assert len(converted) == 1

    def test_foreign_keys_reported_unmatched(self):
        sd = {"totally.unrelated.weight": _FakeTensor()}
        converted, unmatched, _ = _convert_lora_state_dict(sd, {})
        assert converted == {}
        assert unmatched == ["totally.unrelated.weight"]


class TestLoraFailureMessage:
    def test_missing_peft_named_explicitly(self):
        """The #1 field failure: diffusers raising for a missing peft install."""
        exc = ValueError("PEFT backend is required for this method.")
        msg = _lora_failure_message("style_anime", exc)
        assert "peft" in msg.lower()
        assert "style_anime" in msg
        assert "install" in msg.lower()

    def test_other_errors_surface_real_cause(self):
        exc = OSError("No such file or directory")
        msg = _lora_failure_message("missing", exc)
        assert "missing" in msg
        assert "OSError" in msg
        assert "No such file or directory" in msg
