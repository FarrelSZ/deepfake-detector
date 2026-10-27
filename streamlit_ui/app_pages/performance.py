"""Halaman performa model: metrik uji vs target, perbandingan model, kurva training, dan tuning."""

import pandas as pd
import streamlit as st
from ui_utils import charts
from ui_utils.api import ApiClient, ApiError


@st.cache_data(ttl=60, show_spinner="Memuat informasi model...")
def load_model_info(api_url: str) -> dict:
    return ApiClient(api_url).model_info()


st.title("Performa model")

try:
    info = load_model_info(st.session_state.api_url)
except ApiError as err:
    st.error(err, icon=":material/error:")
    st.stop()

test_metrics = info["test_metrics"]
st.caption(
    f"Model: {info['model_name']}. Input {info['img_size']} x {info['img_size']} piksel. "
    f"Threshold keputusan {info['threshold']:.2f}. Kelas positif: fake."
)

tab_test, tab_curves, tab_tuning = st.tabs(["Hasil uji", "Kurva training", "Tuning"])

with tab_test:
    if test_metrics:
        with st.container(horizontal=True):
            for name in charts.METRICS:
                if name in test_metrics:
                    st.metric(name, f"{test_metrics[name]:.3f}", border=True)
        st.altair_chart(
            charts.metrics_vs_target_chart(test_metrics),
            alt="Skor accuracy, precision, recall, F1, dan AUC pada data test vs target minimum",
        )
        st.caption("Batang = hasil pada data test. Garis oranye = target minimum proyek.")
    else:
        st.info("Metrik uji belum tersedia di model_config.json.")

    if info["model_comparison"]:
        st.subheader("Perbandingan model")
        st.altair_chart(
            charts.model_comparison_chart(info["model_comparison"]),
            alt="Perbandingan metrik uji antara CNN baseline dan CNN dengan MHSA",
        )

with tab_curves:
    history = info["training_history"]
    metrics = charts.available_curve_metrics(history) if history else []
    if metrics:
        chosen = st.segmented_control("Metrik", metrics, default=metrics[0], key="curve_metric")
        st.altair_chart(
            charts.training_curves_chart(history, chosen or metrics[0]),
            alt="Kurva metrik training dan validasi per epoch untuk setiap model",
        )
    else:
        st.info("training_history.csv belum ada di folder artifacts/.")

with tab_tuning:
    if info["hyperparameters"]:
        st.subheader("Hyperparameter model final")
        st.json(info["hyperparameters"])
    if info["tuning_results"]:
        st.subheader("Hasil random search")
        st.dataframe(
            pd.DataFrame(info["tuning_results"]),
            hide_index=True,
            alt="Konfigurasi hyperparameter yang dicoba beserta skor validasinya",
        )
    elif not info["hyperparameters"]:
        st.info("Hasil tuning belum tersedia.")
