"""Entrypoint Streamlit: konfigurasi halaman, status API di sidebar, dan navigasi."""

import os

import streamlit as st
from ui_utils.api import ApiClient, ApiError

st.set_page_config(page_title="Deepfake face detector", page_icon=":material/face:", layout="wide")
st.session_state.setdefault("api_url", os.getenv("API_URL", "http://localhost:8000"))


@st.cache_data(ttl=10, show_spinner=False)
def check_api(api_url: str) -> dict:
    """Cek koneksi API. Hasil gagal pun di-cache singkat agar UI tidak menunggu timeout berulang."""
    try:
        return {"ok": True, **ApiClient(api_url, timeout=5).health()}
    except ApiError as err:
        return {"ok": False, "error": str(err)}


with st.sidebar:
    st.header("Pengaturan")
    st.text_input(
        "URL API", key="api_url", help="Alamat server FastAPI, misalnya http://localhost:8000"
    )
    status = check_api(st.session_state.api_url)
    if status["ok"]:
        st.success(f"API terhubung. Model: {status['model_name']}", icon=":material/check_circle:")
    else:
        st.error(status["error"], icon=":material/error:")

if status.get("model_name", "").startswith("dummy"):
    st.warning(
        "Model yang aktif adalah model DUMMY (belum dilatih). Skor prediksi acak dan tak bermakna. "
        "Salin hasil Bagian 9 notebook ke folder artifacts/ lalu restart API.",
        icon=":material/warning:",
    )

navigation = st.navigation(
    [
        st.Page("app_pages/detect.py", title="Deteksi", icon=":material/search:", default=True),
        st.Page("app_pages/performance.py", title="Performa model", icon=":material/bar_chart:"),
        st.Page("app_pages/history.py", title="Riwayat prediksi", icon=":material/history:"),
    ]
)
navigation.run()
