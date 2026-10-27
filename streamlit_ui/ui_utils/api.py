"""Klien HTTP tipis untuk REST API deteksi deepfake."""

from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    """Kesalahan saat memanggil API. Pesannya sudah ramah untuk ditampilkan ke pengguna."""


def _error_detail(response: Any) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, str):
        return detail
    if detail:
        return f"Permintaan tidak valid: {detail}"
    return f"API mengembalikan status {response.status_code}."


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = requests.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise ApiError(
                f"Tidak dapat terhubung ke API di {self.base_url}. Pastikan server API berjalan."
            ) from exc
        if response.status_code >= 400:
            raise ApiError(_error_detail(response))
        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def model_info(self) -> dict:
        return self._request("GET", "/model-info")

    def predict(
        self, filename: str, content: bytes, content_type: str, explain: bool = True
    ) -> dict:
        files = {"file": (filename, content, content_type or "application/octet-stream")}
        return self._request(
            "POST", "/predict", files=files, params={"explain": str(explain).lower()}
        )

    def predictions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        return self._request("GET", "/predictions", params={"limit": limit, "offset": offset})

    def summary(self) -> dict:
        return self._request("GET", "/predictions/summary")
