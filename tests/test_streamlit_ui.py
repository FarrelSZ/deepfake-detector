"""Uji UI end-to-end: AppTest menjalankan Streamlit dan request HTTP-nya diarahkan ke API uji."""

from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ENTRYPOINT = str(Path(__file__).resolve().parents[1] / "streamlit_ui" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def api_bridge(client, monkeypatch):
    """Arahkan requests.request milik klien UI ke aplikasi FastAPI uji (tanpa server nyata)."""

    def forward(method, url, **kwargs):
        kwargs.pop("timeout", None)
        return client.request(method, urlsplit(url).path, **kwargs)

    monkeypatch.setattr(requests, "request", forward)


def run_app() -> AppTest:
    return AppTest.from_file(ENTRYPOINT, default_timeout=60).run()


def test_sidebar_shows_connected_api_and_dummy_warning(api_bridge):
    at = run_app()
    assert not at.exception
    assert any("API terhubung" in success.value for success in at.sidebar.success)
    assert any("DUMMY" in warning.value for warning in at.warning)


def test_api_down_is_reported_without_crashing(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("ditolak")

    monkeypatch.setattr(requests, "request", refuse)
    at = run_app()
    assert not at.exception
    assert any("Tidak dapat terhubung" in error.value for error in at.sidebar.error)


def test_upload_and_analyze_shows_result_and_saves_to_database(api_bridge, client, image_bytes):
    at = run_app()
    at.file_uploader[0].upload("wajah.jpg", image_bytes(), "image/jpeg")
    at.run()
    assert at.button[0].disabled is False

    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    assert any(metric.label == "Skor P(fake)" for metric in at.metric)
    assert client.get("/predictions/summary").json()["total"] == 1

    at.toggle[0].set_value(False).run()  # mengganti tampilan tidak boleh memanggil API lagi
    assert client.get("/predictions/summary").json()["total"] == 1


def test_invalid_upload_shows_api_error(api_bridge):
    at = run_app()
    at.file_uploader[0].upload("rusak.jpg", b"bukan gambar", "image/jpeg")
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert any("bukan gambar" in error.value for error in at.error)


def test_performance_page_renders_metrics_and_curves(api_bridge):
    at = run_app()
    at.switch_page("app_pages/performance.py").run()
    assert not at.exception
    assert not at.error
    assert len(at.metric) == 5


def test_history_page_lists_saved_predictions(api_bridge, client, image_bytes):
    client.post("/predict", files={"file": ("a.jpg", image_bytes(), "image/jpeg")})
    at = run_app()
    at.switch_page("app_pages/history.py").run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 1


def test_history_page_handles_empty_database(api_bridge):
    at = run_app()
    at.switch_page("app_pages/history.py").run()
    assert not at.exception
    assert any("Belum ada prediksi" in info.value for info in at.info)
