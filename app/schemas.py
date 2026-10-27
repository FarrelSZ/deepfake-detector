"""Skema request/response API (Pydantic v2)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class PredictionRecord(ApiModel):
    """Satu prediksi yang tersimpan di database."""

    id: int
    created_at: datetime = Field(description="Waktu UTC")
    filename: str
    image_sha256: str = Field(description="Hash isi file; gambar tidak disimpan")
    label: str = Field(description="'real' atau 'fake'")
    p_fake: float = Field(ge=0, le=1, description="Skor model bahwa gambar palsu")
    threshold: float = Field(description="Label 'fake' jika p_fake >= threshold")
    model_name: str
    latency_ms: float


class PredictionOut(PredictionRecord):
    """Respons POST /predict: rekaman prediksi + Grad-CAM opsional."""

    gradcam_png_base64: str | None = Field(
        default=None, description="Overlay Grad-CAM (PNG, base64) jika explain=true"
    )


class PredictionSummary(ApiModel):
    total: int
    fake: int
    real: int
    mean_p_fake: float | None
    mean_latency_ms: float | None


class HealthOut(ApiModel):
    status: str
    model_loaded: bool
    model_name: str


class ModelInfoOut(ApiModel):
    model_name: str
    dataset: str | None = None
    img_size: int
    class_names: list[str]
    threshold: float
    explain_available: bool
    hyperparameters: dict[str, Any] = {}
    test_metrics: dict[str, float] = {}
    model_comparison: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    tuning_results: list[dict[str, Any]] = []
