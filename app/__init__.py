"""Layanan REST API untuk deteksi wajah palsu (deepfake)."""

import os

# Kurangi log informasi TensorFlow sebelum modul lain mengimpornya.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
