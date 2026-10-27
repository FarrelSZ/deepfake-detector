# Deepfake face detector: API dan Streamlit

Layanan deteksi wajah palsu (real vs fake) dari model **CNN + Multi-Head Self-Attention** yang dilatih di
notebook `Deepfake_Detection_Final_Project.ipynb`. Repo ini mengerjakan **langkah 6 (Deployment)** dan
**langkah 7 (Streamlit UI)** pada guideline Final Project DSML Dibimbing.

| Syarat guideline | Dikerjakan di |
|---|---|
| Layanan REST API dengan FastAPI | `app/` |
| Hasil prediksi disimpan ke database | `app/database.py` (SQLite, PostgreSQL via `DATABASE_URL`) |
| Private repository GitHub dengan CI/CD GitHub Actions | `.github/workflows/ci.yml` |
| Halaman unggah data/input | `streamlit_ui/app_pages/detect.py` |
| Prediksi tampil real-time | `streamlit_ui/app_pages/detect.py` |
| Visualisasi performa model | `streamlit_ui/app_pages/performance.py` dan `history.py` |

## Arsitektur

```
Streamlit UI  --HTTP-->  FastAPI  -->  model Keras (CNN + MHSA, Grad-CAM)
(streamlit_ui)           (app)    -->  database (SQLite / PostgreSQL)
                                  <--  artifacts/ (model, model_config.json, CSV hasil notebook)
```

## Struktur folder

```
app/                 API: main.py (endpoint), predictor.py (model + Grad-CAM), database.py, schemas.py, config.py
streamlit_ui/        UI: streamlit_app.py (entrypoint), app_pages/ (3 halaman), ui_utils/ (klien API, grafik)
artifacts/           Taruh hasil Bagian 9 notebook di sini
scripts/             make_dummy_model.py (model uji), smoke_test.py (dipakai CI)
tests/               pytest: API, predictor, grafik, dan UI end-to-end
.github/workflows/   ci.yml
```

## Menjalankan secara lokal

Prasyarat: Python 3.14

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

**Siapkan artefak model** (pilih salah satu):

- **Model asli:** salin `deepfake_detector.keras`, `model_config.json`, `training_history.csv`,
  `test_metrics.csv`, dan `tuning_results.csv` dari output notebook (Bagian 9) ke folder `artifacts/`.
  Pasang versi TensorFlow yang sama dengan nilai `tensorflow_version` di `model_config.json`.
- **Model dummy (hanya untuk mencoba alur):** `python -m scripts.make_dummy_model --output artifacts`.
  Prediksinya acak. UI menampilkan peringatan selama model dummy aktif.

**Jalankan API**, lalu buka http://localhost:8000/docs:

```bash
uvicorn app.main:app --reload
```

**Jalankan UI** di terminal lain (set `API_URL` bila API tidak di `http://localhost:8000`):

```bash
streamlit run streamlit_ui/streamlit_app.py
```

## Endpoint

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | Status layanan dan nama model |
| GET | `/model-info` | Konfigurasi model, metrik uji, riwayat training, hasil tuning |
| POST | `/predict?explain=true` | Unggah gambar (`file`), terima label, `p_fake`, dan Grad-CAM (PNG base64) |
| GET | `/predictions?limit=50&offset=0` | Riwayat prediksi dari database, terbaru dulu |
| GET | `/predictions/summary` | Total, jumlah real/fake, rata-rata skor dan waktu proses |

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@wajah.jpg;type=image/jpeg"
```

Label `fake` jika `p_fake >= threshold`. Threshold dibaca dari `model_config.json` (dipilih dari data validasi).

## Database

Tabel `predictions`: `id`, `created_at` (UTC), `filename`, `image_sha256`, `label`, `p_fake`, `threshold`,
`model_name`, `latency_ms`. Tabel dibuat otomatis saat API start. Untuk PostgreSQL, pasang `psycopg2-binary`
lalu set `DATABASE_URL=postgresql+psycopg2://user:password@host:5432/dbname`.

## Konfigurasi (environment variable)

| Variabel | Default | Keterangan |
|---|---|---|
| `ARTIFACTS_DIR` | `artifacts` | Folder artefak model |
| `MODEL_PATH` / `MODEL_CONFIG_PATH` | di dalam `ARTIFACTS_DIR` | Lokasi model dan konfigurasi bila berbeda |
| `DATABASE_URL` | `sqlite:///predictions.db` | Koneksi database |
| `MAX_UPLOAD_MB` | `10` | Batas ukuran unggahan |
| `API_URL` | `http://localhost:8000` | Alamat API yang dipakai Streamlit |

## Test dan lint

```bash
pytest -q          # API, predictor, grafik, dan UI end-to-end (Streamlit AppTest terhubung ke API uji)
ruff check .
```

Test memakai model dummy, jadi tidak membutuhkan model asli maupun GPU.

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` punya dua job:

1. **test** (setiap push dan pull request): pasang dependency, `ruff check`, `pytest`.
2. **deploy-check** (push ke `main`, setelah `test` lulus): menjalankan API sungguhan dengan `uvicorn`,
   menjalankan `scripts/smoke_test.py` (health, model-info, predict, tersimpan di database), lalu membuat
   `release.zip` sebagai artifact workflow. Job ini memakai model di `artifacts/` bila ada, atau model dummy
   dengan peringatan bila belum ada.

**Membuat private repository dan push:**

```bash
git init && git add . && git commit -m "Deepfake detector: API dan Streamlit"
gh repo create deepfake-detector --private --source=. --push
```

Tanpa GitHub CLI: buat repository **Private** di github.com, lalu `git remote add origin <url>` dan `git push -u origin main`.
Batas file GitHub adalah 100 MB. Bila model lebih besar, pakai Git LFS.

## Privasi dan keamanan

- Gambar **tidak disimpan**. Yang tersimpan hanya hash SHA-256 dan hasil prediksi. Nama file ikut tersimpan, jadi
  hindari mengunggah file bernama data pribadi.
- Foto wajah termasuk data pribadi. Pastikan ada persetujuan sebelum dipakai pada foto orang lain
  (lihat UU Pelindungan Data Pribadi).
- Unggahan divalidasi: tipe (JPEG/PNG/WebP), ukuran, resolusi, dan isi gambar diperiksa, bukan hanya ekstensi.
- API belum memiliki autentikasi dan rate limiting. Tambahkan keduanya sebelum diakses dari internet.

## Keterbatasan

- Model dilatih pada wajah asli **FFHQ vs wajah sintetis StyleGAN**. Belum teruji pada generator lain
  (misalnya diffusion), video, atau face-swap.
- Input harus berupa **foto wajah tunggal yang ter-align**. Tidak ada deteksi atau crop wajah otomatis, sehingga
  foto seluruh badan atau banyak wajah dapat menghasilkan skor yang tidak andal.
- `p_fake` adalah skor model dan belum tentu probabilitas terkalibrasi. Pakai sebagai skrining dengan review manusia.

## Troubleshooting

- **`Gagal memuat model`:** versi TensorFlow/Keras berbeda dari saat training. Samakan dengan `model_config.json`.
- **UI menampilkan "Tidak dapat terhubung ke API":** pastikan `uvicorn` berjalan dan `URL API` di sidebar benar.
- **UI menampilkan peringatan model DUMMY:** `artifacts/` masih berisi model uji. Ganti dengan hasil notebook lalu restart API.
