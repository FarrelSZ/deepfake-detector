"""Aplikasi FastAPI: endpoint prediksi, riwayat (database), dan informasi model."""

import base64
import hashlib
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.artifacts import read_csv_records
from app.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_IMAGE_FORMATS,
    MAX_IMAGE_PIXELS,
    Settings,
)
from app.database import (
    create_session_factory,
    list_predictions,
    save_prediction,
    summarize_predictions,
)
from app.predictor import DeepfakePredictor
from app.schemas import (
    HealthOut,
    ModelInfoOut,
    PredictionOut,
    PredictionRecord,
    PredictionSummary,
)

router = APIRouter()


# ------------------------------------------------------------------ dependency
def get_predictor(request: Request) -> DeepfakePredictor:
    return request.app.state.predictor


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


PredictorDep = Annotated[DeepfakePredictor, Depends(get_predictor)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[Session, Depends(get_session)]


# ------------------------------------------------------------------ helper upload
def decode_upload(upload: UploadFile, max_bytes: int) -> tuple[bytes, Image.Image]:
    """Validasi file unggahan (tipe, ukuran, isi) lalu kembalikan byte asli dan gambar RGB."""
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail="Tipe file tidak didukung. Gunakan JPEG, PNG, atau WebP."
        )

    data = upload.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="File kosong.")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Ukuran file melebihi batas {max_bytes // (1024 * 1024)} MB."
        )

    try:
        image = Image.open(BytesIO(data))
        if image.format not in ALLOWED_IMAGE_FORMATS:
            raise HTTPException(status_code=415, detail="Format gambar tidak didukung.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Resolusi gambar terlalu besar.")
        image.load()  # decode penuh agar file rusak ketahuan di sini
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.") from None

    return data, ImageOps.exif_transpose(image).convert("RGB")


def encode_png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ------------------------------------------------------------------ endpoint
@router.get("/health", response_model=HealthOut, tags=["sistem"])
def health(predictor: PredictorDep) -> HealthOut:
    return HealthOut(status="ok", model_loaded=True, model_name=predictor.model_name)


@router.get("/model-info", response_model=ModelInfoOut, tags=["sistem"])
def model_info(predictor: PredictorDep, settings: SettingsDep) -> ModelInfoOut:
    """Konfigurasi model dan artefak performa (untuk halaman 'Performa model' di Streamlit)."""
    config, artifacts = predictor.config, settings.artifacts_dir
    return ModelInfoOut(
        model_name=predictor.model_name,
        dataset=config.get("dataset"),
        img_size=predictor.img_size,
        class_names=config.get("class_names", ["real", "fake"]),
        threshold=predictor.threshold,
        explain_available=predictor.can_explain,
        hyperparameters=config.get("hyperparameters", {}),
        test_metrics=config.get("test_metrics", {}),
        model_comparison=read_csv_records(artifacts / "test_metrics.csv"),
        training_history=read_csv_records(artifacts / "training_history.csv"),
        tuning_results=read_csv_records(artifacts / "tuning_results.csv"),
    )


@router.post("/predict", response_model=PredictionOut, tags=["prediksi"])
def predict(
    file: UploadFile,
    predictor: PredictorDep,
    session: SessionDep,
    settings: SettingsDep,
    explain: bool = False,
) -> PredictionOut:
    """Klasifikasi satu foto wajah (real vs fake) dan simpan hasilnya ke database."""
    data, image = decode_upload(file, settings.max_upload_bytes)

    started = time.perf_counter()
    result = predictor.predict(image)
    latency_ms = (time.perf_counter() - started) * 1000

    record = save_prediction(
        session,
        filename=Path(file.filename or "upload").name[:120],
        image_sha256=hashlib.sha256(data).hexdigest(),
        label=result.label,
        p_fake=result.p_fake,
        threshold=predictor.threshold,
        model_name=predictor.model_name,
        latency_ms=latency_ms,
    )
    gradcam = (
        encode_png_base64(predictor.explain(image)) if explain and predictor.can_explain else None
    )
    return PredictionOut(
        **PredictionRecord.model_validate(record).model_dump(), gradcam_png_base64=gradcam
    )


@router.get("/predictions", response_model=list[PredictionRecord], tags=["prediksi"])
def predictions(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PredictionRecord]:
    """Riwayat prediksi dari database, terbaru lebih dulu."""
    return list_predictions(session, limit=limit, offset=offset)


@router.get("/predictions/summary", response_model=PredictionSummary, tags=["prediksi"])
def predictions_summary(session: SessionDep) -> PredictionSummary:
    return PredictionSummary(**summarize_predictions(session))


# ------------------------------------------------------------------ factory
def create_app(
    settings: Settings | None = None, predictor: DeepfakePredictor | None = None
) -> FastAPI:
    """Bangun aplikasi. Parameter predictor memungkinkan test memakai model yang sudah dimuat."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or DeepfakePredictor(
            settings.model_path, settings.config_path
        )
        app.state.session_factory = create_session_factory(settings.database_url)
        yield
        app.state.session_factory.kw["bind"].dispose()

    app = FastAPI(
        title="Deepfake face detector API",
        description="Klasifikasi foto wajah real vs fake (CNN + Multi-Head Self-Attention).",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(router)
    return app


app = create_app()
