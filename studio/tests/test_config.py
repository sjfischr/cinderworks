"""Tests for the Config module."""

from pathlib import Path

from studio.config import Config, _STUDIO_ROOT


def test_config_app_name_default():
    """APP_NAME defaults to 'Cinderworks' when no .env override."""
    assert Config.APP_NAME == "Cinderworks"


def test_config_paths_are_pathlib():
    """All path attributes use pathlib.Path (Requirement 14.1)."""
    assert isinstance(Config.MODEL_DIR, Path)
    assert isinstance(Config.OUTPUT_DIR, Path)
    assert isinstance(Config.DB_PATH, Path)
    assert isinstance(Config.BASE_DIR, Path)


def test_config_default_model_dir():
    """MODEL_DIR defaults to models_store under studio root."""
    assert Config.MODEL_DIR == _STUDIO_ROOT / "models_store"


def test_config_default_output_dir():
    """OUTPUT_DIR defaults to outputs under studio root."""
    assert Config.OUTPUT_DIR == _STUDIO_ROOT / "outputs"


def test_config_default_db_path():
    """DB_PATH defaults to studio.db under studio root."""
    assert Config.DB_PATH == _STUDIO_ROOT / "studio.db"


def test_config_ensure_dirs(tmp_path, monkeypatch):
    """ensure_dirs creates all required directories."""
    monkeypatch.setattr(Config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(Config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(Config, "DB_PATH", tmp_path / "data" / "studio.db")

    Config.ensure_dirs()

    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "data").is_dir()
