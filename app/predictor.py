"""Memuat model Keras dan menjalankan inferensi dengan preprocessing yang sama seperti training."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from app.config import LABEL_FAKE, LABEL_REAL

REQUIRED_CONFIG_KEYS = ("model_name", "img_size", "threshold")


@dataclass(frozen=True)
class PredictionResult:
    """Hasil klasifikasi satu gambar."""

    label: str
    p_fake: float


def load_model_config(config_path: Path) -> dict:
    """Baca model_config.json (dibuat di Bagian 9 notebook) dan pastikan kunci wajibnya ada."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Konfigurasi model tidak ditemukan: {config_path}. "
            "Salin model_config.json dari notebook (Bagian 9) ke folder artifacts/."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"model_config.json tidak lengkap, kunci yang hilang: {missing}")
    return config


def apply_jet_colormap(values: np.ndarray) -> np.ndarray:
    """Petakan nilai 0-1 menjadi warna RGB (uint8) bergaya 'jet'."""
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype("uint8")


def overlay_heatmap(
    image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45, max_side: int = 512
) -> Image.Image:
    """Tempelkan heatmap Grad-CAM (nilai 0-1) di atas gambar asli."""
    base = image.convert("RGB")
    base.thumbnail((max_side, max_side))
    cam = Image.fromarray((heatmap * 255).astype("uint8")).resize(
        base.size, Image.Resampling.BILINEAR
    )
    colored = Image.fromarray(apply_jet_colormap(np.asarray(cam) / 255.0))
    return Image.blend(base, colored, alpha)


class DeepfakePredictor:
    """Membungkus model dan konfigurasinya, sehingga API cukup memanggil predict() dan explain()."""

    def __init__(self, model_path: Path, config_path: Path) -> None:
        self.config = load_model_config(config_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}. "
                "Salin deepfake_detector.keras dari notebook (Bagian 9) ke folder artifacts/."
            )
        try:
            self._model = keras.models.load_model(model_path, compile=False)
        except Exception as exc:
            trained_with = self.config.get("tensorflow_version", "lihat model_config.json")
            raise RuntimeError(
                f"Gagal memuat model {model_path}. Gunakan versi TensorFlow yang sama "
                f"dengan saat training ({trained_with})."
            ) from exc

        self.model_name: str = self.config["model_name"]
        self.img_size = int(self.config["img_size"])
        self.threshold = float(self.config["threshold"])
        self._grad_model = self._build_grad_model(self.config.get("gradcam_layer"))
        self._lock = threading.Lock()

    @property
    def can_explain(self) -> bool:
        return self._grad_model is not None

    def _build_grad_model(self, layer_name: str | None) -> keras.Model | None:
        """Model yang mengembalikan feature map konvolusi terakhir dan prediksi (untuk Grad-CAM)."""
        if not layer_name:
            return None
        try:
            layer = self._model.get_layer(layer_name)
        except ValueError:
            return None
        return keras.Model(self._model.inputs, [layer.output, self._model.output])

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize persis seperti saat training (bilinear + antialias). Piksel tetap 0-255."""
        rgb = np.asarray(image.convert("RGB"))
        resized = tf.image.resize(
            rgb, (self.img_size, self.img_size), method="bilinear", antialias=True
        )
        return resized.numpy()[None].astype("float32")

    def predict(self, image: Image.Image) -> PredictionResult:
        batch = self.preprocess(image)
        with self._lock:
            p_fake = float(self._model(batch, training=False).numpy()[0, 0])
        label = LABEL_FAKE if p_fake >= self.threshold else LABEL_REAL
        return PredictionResult(label=label, p_fake=p_fake)

    def explain(self, image: Image.Image) -> Image.Image:
        """Grad-CAM terhadap skor P(fake): area terang paling mendorong keputusan ke arah FAKE."""
        if self._grad_model is None:
            raise RuntimeError("Grad-CAM tidak tersedia untuk model ini.")
        batch = tf.convert_to_tensor(self.preprocess(image))
        with self._lock, tf.GradientTape() as tape:
            features, predictions = self._grad_model(batch, training=False)
            score = predictions[:, 0]
        gradients = tape.gradient(score, features)
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))
        cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
        return overlay_heatmap(image, cam.numpy()[0])
