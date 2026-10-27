import json

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from app.predictor import DeepfakePredictor, load_model_config, overlay_heatmap


def test_preprocess_matches_training_pipeline(shared_predictor):
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (80, 60, 3), dtype="uint8"))
    batch = shared_predictor.preprocess(image)

    size = shared_predictor.img_size
    expected = tf.image.resize(np.asarray(image), (size, size), method="bilinear", antialias=True)
    assert batch.shape == (1, size, size, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch[0], expected.numpy(), rtol=1e-5)
    assert batch.max() <= 255


def test_preprocess_converts_non_rgb_modes(shared_predictor):
    batch = shared_predictor.preprocess(Image.new("L", (40, 40), 128))
    assert batch.shape[-1] == 3


@pytest.mark.parametrize(("threshold", "expected"), [(0.0, "fake"), (1.01, "real")])
def test_label_follows_threshold(shared_predictor, monkeypatch, threshold, expected):
    monkeypatch.setattr(shared_predictor, "threshold", threshold)
    result = shared_predictor.predict(Image.new("RGB", (64, 64), (90, 90, 90)))
    assert 0 <= result.p_fake <= 1
    assert result.label == expected


def test_explain_returns_overlay_within_size_limit(shared_predictor):
    assert shared_predictor.can_explain
    overlay = shared_predictor.explain(Image.new("RGB", (800, 600), (200, 100, 50)))
    assert overlay.mode == "RGB"
    assert max(overlay.size) <= 512


def test_overlay_heatmap_keeps_aspect_ratio():
    overlay = overlay_heatmap(Image.new("RGB", (200, 100)), np.ones((8, 8)))
    assert overlay.size == (200, 100)


def test_missing_files_give_actionable_errors(tmp_path, dummy_artifacts):
    with pytest.raises(FileNotFoundError, match="model_config.json"):
        load_model_config(tmp_path / "model_config.json")
    with pytest.raises(FileNotFoundError, match="deepfake_detector.keras"):
        DeepfakePredictor(
            tmp_path / "deepfake_detector.keras", dummy_artifacts / "model_config.json"
        )


def test_incomplete_config_is_rejected(tmp_path):
    path = tmp_path / "model_config.json"
    path.write_text(json.dumps({"model_name": "x"}))
    with pytest.raises(ValueError, match="img_size"):
        load_model_config(path)


def test_corrupt_model_reports_tensorflow_version(tmp_path, dummy_artifacts):
    broken = tmp_path / "deepfake_detector.keras"
    broken.write_bytes(b"bukan model")
    with pytest.raises(RuntimeError, match="versi TensorFlow"):
        DeepfakePredictor(broken, dummy_artifacts / "model_config.json")
