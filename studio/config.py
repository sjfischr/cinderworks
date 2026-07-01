"""Central configuration for Cinderworks Studio.

All paths resolve off a single base directory. No path literals scattered
through the codebase — everything goes through this Config object.
"""

from pathlib import Path

from dotenv import load_dotenv
import os


# Load .env from the studio root (same directory as this file)
_STUDIO_ROOT = Path(__file__).resolve().parent
load_dotenv(_STUDIO_ROOT / ".env")


class Config:
    """Resolves all application paths and settings from environment variables.

    Defaults are relative to the studio root so the app works out of the box
    without a .env file (dev convenience), but production usage should always
    provide an explicit .env.
    """

    APP_NAME: str = os.getenv("APP_NAME", "Cinderworks")

    # Paths — all resolved via pathlib for platform-agnostic handling
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", str(_STUDIO_ROOT / "models_store")))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(_STUDIO_ROOT / "outputs")))
    DB_PATH: Path = Path(os.getenv("DB_PATH", str(_STUDIO_ROOT / "studio.db")))

    # Convenience: the studio root itself
    BASE_DIR: Path = _STUDIO_ROOT

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
