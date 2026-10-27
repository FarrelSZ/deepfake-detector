"""Penyimpanan hasil prediksi dengan SQLAlchemy (SQLite default, PostgreSQL via DATABASE_URL)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, String, case, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import LABEL_FAKE


def utc_now() -> datetime:
    """Waktu UTC tanpa tzinfo, agar konsisten di SQLite maupun PostgreSQL."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Satu baris per gambar yang diprediksi. Gambar itu sendiri tidak disimpan, hanya hash-nya."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(16), index=True)
    p_fake: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    """Buat engine, buat tabel bila belum ada, lalu kembalikan pabrik session."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def save_prediction(session: Session, **fields) -> Prediction:
    record = Prediction(**fields)
    session.add(record)
    session.commit()
    return record


def list_predictions(session: Session, limit: int, offset: int) -> list[Prediction]:
    """Riwayat prediksi, terbaru lebih dulu."""
    query = (
        select(Prediction)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(session.scalars(query))


def summarize_predictions(session: Session) -> dict:
    """Agregat sederhana untuk dashboard monitoring."""
    fake_count = func.coalesce(func.sum(case((Prediction.label == LABEL_FAKE, 1), else_=0)), 0)
    total, fake, mean_p_fake, mean_latency = session.execute(
        select(
            func.count(Prediction.id),
            fake_count,
            func.avg(Prediction.p_fake),
            func.avg(Prediction.latency_ms),
        )
    ).one()
    return {
        "total": total,
        "fake": int(fake),
        "real": total - int(fake),
        "mean_p_fake": mean_p_fake,
        "mean_latency_ms": mean_latency,
    }
