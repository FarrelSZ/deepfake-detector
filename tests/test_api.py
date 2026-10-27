import base64
import re
from io import BytesIO

import pytest
from PIL import Image


def upload(
    client, content: bytes, content_type: str = "image/jpeg", name: str = "wajah.jpg", **params
):
    return client.post("/predict", files={"file": (name, content, content_type)}, params=params)


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_name": "dummy_untrained"}


def test_model_info_exposes_config_and_artifacts(client):
    body = client.get("/model-info").json()
    assert body["img_size"] == 32
    assert body["explain_available"] is True
    assert set(body["test_metrics"]) >= {"accuracy", "auc"}
    assert body["model_comparison"][0]["name"].startswith("CNN baseline")
    assert {"model", "epoch", "val_auc"} <= set(body["training_history"][0])
    assert len(body["tuning_results"]) == 1


@pytest.mark.parametrize(("fmt", "content_type"), [("JPEG", "image/jpeg"), ("PNG", "image/png")])
def test_predict_returns_score_and_persists(client, image_bytes, fmt, content_type):
    response = upload(client, image_bytes(fmt), content_type)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"real", "fake"}
    assert 0 <= body["p_fake"] <= 1
    assert body["gradcam_png_base64"] is None
    assert body["filename"] == "wajah.jpg"
    assert re.fullmatch(r"[0-9a-f]{64}", body["image_sha256"])
    assert client.get("/predictions").json()[0]["id"] == body["id"]


def test_predict_with_explain_returns_png(client, image_bytes):
    body = upload(client, image_bytes(), explain="true").json()
    overlay = Image.open(BytesIO(base64.b64decode(body["gradcam_png_base64"])))
    assert overlay.format == "PNG"


def test_image_content_is_not_stored(client, image_bytes):
    upload(client, image_bytes())
    stored = client.get("/predictions").json()[0]
    assert set(stored) == {
        "id",
        "created_at",
        "filename",
        "image_sha256",
        "label",
        "p_fake",
        "threshold",
        "model_name",
        "latency_ms",
    }


def test_filename_is_reduced_to_basename(client, image_bytes):
    body = upload(client, image_bytes(), name="../../etc/foto_ktp.jpg").json()
    assert body["filename"] == "foto_ktp.jpg"


def test_rejects_unsupported_content_type(client):
    response = upload(client, b"hello", "text/plain", "catatan.txt")
    assert response.status_code == 415


def test_rejects_corrupt_image(client):
    response = upload(client, b"ini bukan gambar", "image/jpeg")
    assert response.status_code == 400
    assert "valid" in response.json()["detail"]


def test_rejects_empty_file(client):
    assert upload(client, b"").status_code == 400


def test_rejects_declared_jpeg_that_is_really_gif(client):
    buffer = BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, format="GIF")
    assert upload(client, buffer.getvalue(), "image/jpeg").status_code == 415


def test_rejects_oversized_upload(client):
    assert upload(client, b"x" * 200_001).status_code == 413


def test_history_is_newest_first_and_paginated(client, image_bytes):
    ids = [upload(client, image_bytes(color=(i * 40, 0, 0))).json()["id"] for i in range(3)]

    assert [row["id"] for row in client.get("/predictions").json()] == ids[::-1]
    page = client.get("/predictions", params={"limit": 1, "offset": 1}).json()
    assert [row["id"] for row in page] == [ids[1]]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"offset": -1}])
def test_history_rejects_invalid_pagination(client, params):
    assert client.get("/predictions", params=params).status_code == 422


def test_summary_counts_labels(client, image_bytes):
    assert client.get("/predictions/summary").json()["total"] == 0
    for i in range(3):
        upload(client, image_bytes(color=(i * 50, 10, 10)))

    summary = client.get("/predictions/summary").json()
    assert summary["total"] == 3
    assert summary["fake"] + summary["real"] == 3
    assert 0 <= summary["mean_p_fake"] <= 1
