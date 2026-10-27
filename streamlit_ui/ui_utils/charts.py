"""Pembuat grafik Altair untuk halaman performa dan riwayat. Fungsi murni sehingga mudah diuji."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

REAL_COLOR = "#0BB1CB"
FAKE_COLOR = "#E8641F"
PRIMARY_COLOR = "#0891A8"
LABEL_SCALE = alt.Scale(domain=["real", "fake"], range=[REAL_COLOR, FAKE_COLOR])

# Target minimum yang sama dengan Bagian 1.5 notebook.
TARGETS = {"accuracy": 0.90, "precision": 0.90, "recall": 0.90, "f1": 0.90, "auc": 0.95}
METRICS = list(TARGETS)


def metrics_vs_target_chart(test_metrics: dict[str, float]) -> alt.LayerChart:
    """Batang = skor model pada data test, garis oranye = target minimum."""
    frame = pd.DataFrame(
        {
            "metrik": METRICS,
            "nilai": [test_metrics.get(name) for name in METRICS],
            "target": [TARGETS[name] for name in METRICS],
        }
    )
    x = alt.X("metrik:N", sort=METRICS, title=None)
    bars = (
        alt.Chart(frame)
        .mark_bar(color=PRIMARY_COLOR)
        .encode(
            x=x,
            y=alt.Y("nilai:Q", scale=alt.Scale(domain=[0, 1]), title="Skor"),
            tooltip=["metrik", "nilai", "target"],
        )
    )
    targets = (
        alt.Chart(frame).mark_tick(color=FAKE_COLOR, thickness=3, size=40).encode(x=x, y="target:Q")
    )
    return (bars + targets).properties(height=300)


def model_comparison_chart(comparison: list[dict[str, Any]]) -> alt.Chart:
    """Perbandingan beberapa model/threshold (isi test_metrics.csv) pada tiap metrik."""
    frame = pd.DataFrame(comparison)
    metrics = [name for name in METRICS if name in frame.columns]
    long = frame.melt(id_vars="name", value_vars=metrics, var_name="metrik", value_name="nilai")
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("metrik:N", sort=METRICS, title=None),
            xOffset="name:N",
            y=alt.Y("nilai:Q", scale=alt.Scale(domain=[0, 1]), title="Skor"),
            color=alt.Color("name:N", title="Model"),
            tooltip=["name", "metrik", "nilai"],
        )
        .properties(height=320)
    )


def available_curve_metrics(history: list[dict[str, Any]]) -> list[str]:
    """Metrik yang punya pasangan train dan validasi pada riwayat training."""
    columns = set(pd.DataFrame(history).columns)
    return [
        name
        for name in ("loss", "accuracy", "auc", "precision", "recall")
        if {name, f"val_{name}"} <= columns
    ]


def training_curves_chart(history: list[dict[str, Any]], metric: str) -> alt.Chart:
    """Kurva train (garis putus-putus) vs validasi untuk tiap model."""
    frame = pd.DataFrame(history)
    long = frame.melt(
        id_vars=["model", "epoch"],
        value_vars=[metric, f"val_{metric}"],
        var_name="seri",
        value_name="nilai",
    )
    long["seri"] = long["seri"].map({metric: "train", f"val_{metric}": "validasi"})
    long["epoch"] = long["epoch"] + 1
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("nilai:Q", title=metric, scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Model"),
            strokeDash=alt.StrokeDash("seri:N", title="Data"),
            tooltip=["model", "seri", "epoch", "nilai"],
        )
        .properties(height=320)
    )


def prediction_distribution_chart(records: list[dict[str, Any]]) -> alt.Chart:
    """Histogram skor P(fake) dari riwayat prediksi, diwarnai menurut label."""
    return (
        alt.Chart(pd.DataFrame(records))
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("p_fake:Q", bin=alt.Bin(maxbins=20, extent=[0, 1]), title="P(fake)"),
            y=alt.Y("count()", title="Jumlah prediksi"),
            color=alt.Color("label:N", scale=LABEL_SCALE, title="Label"),
        )
        .properties(height=280)
    )


def label_counts_chart(summary: dict[str, Any]) -> alt.Chart:
    """Jumlah prediksi real vs fake."""
    frame = pd.DataFrame({"label": ["real", "fake"], "jumlah": [summary["real"], summary["fake"]]})
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=["real", "fake"], title=None),
            y=alt.Y("jumlah:Q", title="Jumlah prediksi"),
            color=alt.Color("label:N", scale=LABEL_SCALE, legend=None),
            tooltip=["label", "jumlah"],
        )
        .properties(height=280)
    )
