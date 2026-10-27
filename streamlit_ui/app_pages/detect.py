"""Halaman deteksi: unggah foto wajah, kirim ke API, lalu tampilkan hasil prediksi."""

import base64

import pandas as pd
import streamlit as st
from ui_utils.api import ApiClient, ApiError

ACCEPTED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]


def analyze_upload(upload) -> dict:
    """Kirim satu file ke API. Error dikemas agar satu file gagal tidak menghentikan halaman."""
    content = upload.getvalue()
    try:
        result = ApiClient(st.session_state.api_url).predict(upload.name, content, upload.type)
        return {"name": upload.name, "content": content, "result": result, "error": None}
    except ApiError as err:
        return {"name": upload.name, "content": content, "result": None, "error": str(err)}


def render_result(item: dict, show_gradcam: bool) -> None:
    with st.container(border=True):
        st.markdown(f"**{item['name']}**")
        if item["error"]:
            st.error(item["error"], icon=":material/error:")
            return

        result = item["result"]
        is_fake = result["label"] == "fake"
        gradcam = result.get("gradcam_png_base64") if show_gradcam else None
        columns = st.columns(3 if gradcam else 2)

        columns[0].image(
            item["content"], alt=f"Foto yang diunggah: {item['name']}", width="stretch"
        )
        with columns[1]:
            if is_fake:
                st.badge(
                    "FAKE: terdeteksi wajah sintetis", color="orange", icon=":material/warning:"
                )
            else:
                st.badge("REAL: terdeteksi foto asli", color="green", icon=":material/verified:")
            st.metric("Skor P(fake)", f"{result['p_fake']:.1%}")
            st.progress(result["p_fake"], text=f"Threshold keputusan: {result['threshold']:.2f}")
            st.caption(f"Waktu proses {result['latency_ms']:.0f} ms. ID prediksi #{result['id']}.")
        if gradcam:
            columns[2].image(
                base64.b64decode(gradcam),
                caption="Grad-CAM: area terang paling mendorong skor ke arah FAKE",
                alt="Peta panas Grad-CAM yang ditempel pada foto",
                width="stretch",
            )


st.title("Deteksi wajah palsu")
st.write(
    "Unggah foto wajah untuk mengetahui apakah gambar tersebut foto asli atau buatan AI (StyleGAN)."
)
st.caption(
    "Model dilatih pada foto wajah tunggal yang ter-align (seperti dataset FFHQ). "
    "Foto yang jauh berbeda dari kondisi itu dapat menghasilkan skor yang tidak andal."
)

uploads = st.file_uploader(
    "Unggah foto wajah", type=ACCEPTED_EXTENSIONS, accept_multiple_files=True, key="uploads"
)
show_gradcam = st.toggle(
    "Tampilkan Grad-CAM", value=True, help="Menunjukkan area foto yang memengaruhi skor."
)

# Hasil lama dibuang bila daftar unggahan berubah, supaya tidak tertukar dengan file baru.
signature = tuple(upload.file_id for upload in uploads)
if st.session_state.get("results_signature") != signature:
    st.session_state.results = []
    st.session_state.results_signature = signature

if st.button("Analisis gambar", type="primary", icon=":material/search:", disabled=not uploads):
    with st.spinner("Menganalisis gambar..."):
        st.session_state.results = [analyze_upload(upload) for upload in uploads]

results = st.session_state.results
if len(results) > 1:
    summary = pd.DataFrame(
        {
            "file": [item["name"] for item in results],
            "label": [item["result"]["label"] if item["result"] else "error" for item in results],
            "p_fake": [item["result"]["p_fake"] if item["result"] else None for item in results],
        }
    )
    st.dataframe(summary, hide_index=True, alt="Ringkasan label dan skor P(fake) untuk setiap file")
for item in results:
    render_result(item, show_gradcam)
