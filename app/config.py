"""Konfigurasi aplikasi yang dibaca dari environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LABEL_REAL = "real"
LABEL_FAKE = "fake"

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_PIXELS = 25_000_000


@dataclass(frozen=True)
class Settings:
    """Pengaturan layanan. Nilai default mengikuti struktur folder repo."""

    model_path: Path
    config_path: Path
    database_url: str
    max_upload_bytes: int

    @property
    def artifacts_dir(self) -> Path:
        """Folder artefak: model, konfigurasi, riwayat training, dan hasil tuning."""
        return self.model_path.parent

    @classmethod
    def from_env(cls) -> Settings:
        artifacts = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", artifacts / "deepfake_detector.keras")),
            config_path=Path(os.getenv("MODEL_CONFIG_PATH", artifacts / "model_config.json")),
            database_url=os.getenv("DATABASE_URL", "sqlite:///predictions.db"),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024,
        )
