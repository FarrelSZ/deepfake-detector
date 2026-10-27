import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.predictor import DeepfakePredictor  # noqa: E402
from scripts.make_dummy_model import write_dummy_artifacts  # noqa: E402


@pytest.fixture(scope="session")
def dummy_artifacts(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("artifacts")
    write_dummy_artifacts(directory)
    return directory


@pytest.fixture(scope="session")
def shared_predictor(dummy_artifacts) -> DeepfakePredictor:
    """Model dimuat sekali per sesi test agar suite tetap cepat."""
    return DeepfakePredictor(
        dummy_artifacts / "deepfake_detector.keras", dummy_artifacts / "model_config.json"
    )


@pytest.fixture
def settings(dummy_artifacts, tmp_path) -> Settings:
    return Settings(
        model_path=dummy_artifacts / "deepfake_detector.keras",
        config_path=dummy_artifacts / "model_config.json",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        max_upload_bytes=200_000,
    )


@pytest.fixture
def client(settings, shared_predictor):
    with TestClient(create_app(settings, shared_predictor)) as test_client:
        yield test_client


@pytest.fixture
def image_bytes():
    """Pabrik gambar uji: image_bytes(fmt='PNG', size=(64, 48), color=(10, 20, 30))."""

    def make(fmt: str = "JPEG", size: tuple[int, int] = (96, 80), color=(120, 80, 60)) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", size, color).save(buffer, format=fmt)
        return buffer.getvalue()

    return make
