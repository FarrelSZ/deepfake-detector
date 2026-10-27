"""Halaman riwayat: prediksi yang tersimpan di database beserta ringkasannya."""

import pandas as pd
import streamlit as st
from ui_utils import charts
from ui_utils.api import ApiClient, ApiError

st.title("Riwayat prediksi")
st.caption(
    "Data dibaca dari database API. Gambar tidak disimpan, hanya hasil prediksi dan hash file."
)

limit = st.select_slider(
    "Jumlah riwayat yang ditampilkan", options=[25, 50, 100, 200, 500], value=50
)

try:
    client = ApiClient(st.session_state.api_url)
    summary = client.summary()
    records = client.predictions(limit=limit)
except ApiError as err:
    st.error(err, icon=":material/error:")
    st.stop()

if summary["total"] == 0:
    st.info("Belum ada prediksi. Unggah foto di halaman Deteksi.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Total prediksi", summary["total"], border=True)
    st.metric("Terdeteksi fake", f"{summary['fake'] / summary['total']:.0%}", border=True)
    st.metric("Rata-rata P(fake)", f"{summary['mean_p_fake']:.3f}", border=True)
    st.metric("Rata-rata waktu proses", f"{summary['mean_latency_ms']:.0f} ms", border=True)

left, right = st.columns(2)
left.altair_chart(
    charts.label_counts_chart(summary), alt="Jumlah prediksi berlabel real dibandingkan fake"
)
right.altair_chart(
    charts.prediction_distribution_chart(records),
    alt="Sebaran skor P(fake) pada riwayat prediksi, diwarnai menurut label",
)

table = pd.DataFrame(records)[
    ["id", "created_at", "filename", "label", "p_fake", "latency_ms", "model_name"]
]
st.dataframe(table, hide_index=True, alt="Tabel riwayat prediksi terbaru")
