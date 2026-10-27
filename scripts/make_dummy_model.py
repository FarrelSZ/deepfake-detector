"""Buat artefak model DUMMY (tidak dilatih) untuk mencoba API dan Streamlit tanpa notebook.

Prediksi model dummy acak dan TIDAK bermakna. Ganti seluruh isi folder artifacts/ dengan hasil
Bagian 9 notebook sebelum dipakai untuk demo atau presentasi.

Pemakaian:  python -m scripts.make_dummy_model --output artifacts
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

layers = keras.layers

DUMMY_MODEL_NAME = "dummy_untrained"
DUMMY_THRESHOLD = 0.45
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "auc")
HISTORY_METRICS = ("loss", "accuracy", "precision", "recall", "auc")


def build_dummy_model(img_size: int = 32) -> keras.Model:
    """Model kecil dengan kontrak yang sama seperti model asli: input 0-255, output P(fake)."""
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for block_id, filters in enumerate((4, 8), start=1):
        x = layers.Conv2D(filters, 3, padding="same", name=f"block{block_id}_conv1")(x)
        x = layers.Activation("relu", name=f"block{block_id}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", name=f"block{block_id}_conv2")(x)
        x = layers.Activation("relu", name=f"block{block_id}_relu2")(x)
        x = layers.MaxPooling2D(name=f"block{block_id}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(8, activation="relu", name="embedding")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_fake")(x)
    return keras.Model(inputs, outputs, name=DUMMY_MODEL_NAME)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _dummy_history(rng: np.random.Generator, epochs: int = 5) -> list[list]:
    """Kurva 'tanpa belajar' (loss ~0.69, akurasi ~0.5) dengan format training_history.csv."""
    rows = []
    for model_name in ("CNN baseline", "CNN + MHSA"):
        for epoch in range(epochs):
            train = [0.69 + rng.normal(0, 0.005)] + [0.5 + rng.normal(0, 0.01) for _ in range(4)]
            val = [0.69 + rng.normal(0, 0.005)] + [0.5 + rng.normal(0, 0.01) for _ in range(4)]
            rows.append([model_name, epoch, *train, *val, 1e-3])
    return rows


def write_dummy_artifacts(output_dir: Path, img_size: int = 32, seed: int = 0) -> None:
    """Tulis model + model_config.json + CSV artefak dengan format yang sama seperti notebook."""
    output_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    build_dummy_model(img_size).save(output_dir / "deepfake_detector.keras")

    chance = {name: 0.5 for name in METRIC_NAMES}
    config = {
        "model_name": DUMMY_MODEL_NAME,
        "dataset": "DUMMY - bukan hasil training",
        "img_size": img_size,
        "class_names": ["real", "fake"],
        "positive_class": "fake",
        "threshold": DUMMY_THRESHOLD,
        "input_spec": "RGB float32 skala 0-255, di-resize ke img_size x img_size.",
        "output_spec": "P(fake) pada rentang 0-1. Label fake jika P(fake) >= threshold.",
        "gradcam_layer": "block2_relu2",
        "hyperparameters": {"learning_rate": 0.001, "base_filters": 4, "num_blocks": 2},
        "n_images": {"train": 0, "valid": 0, "test": 0},
        "test_metrics": chance,
        "tensorflow_version": "dummy",
    }
    (output_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history_header = ["model", "epoch", *HISTORY_METRICS, *(f"val_{m}" for m in HISTORY_METRICS)]
    _write_csv(
        output_dir / "training_history.csv", [*history_header, "learning_rate"], _dummy_history(rng)
    )

    comparison_names = [
        "CNN baseline (threshold 0.50)",
        "CNN + MHSA (threshold 0.50)",
        f"CNN + MHSA (threshold {DUMMY_THRESHOLD:.2f})",
    ]
    _write_csv(
        output_dir / "test_metrics.csv",
        ["", *METRIC_NAMES],
        [[name, *chance.values()] for name in comparison_names],
    )
    _write_csv(
        output_dir / "tuning_results.csv",
        [
            "learning_rate",
            "base_filters",
            "num_blocks",
            "num_heads",
            "dropout",
            "batch_size",
            "val_auc",
            "val_accuracy",
            "val_loss",
            "best_epoch",
            "minutes",
        ],
        [[1e-3, 4, 2, 2, 0.3, 32, 0.5, 0.5, 0.69, 1, 0.1]],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    write_dummy_artifacts(args.output)
    print(f"Artefak DUMMY ditulis ke {args.output}. Jangan dipakai untuk demo hasil nyata.")
