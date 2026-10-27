"""Membaca file CSV artefak notebook (riwayat training, perbandingan model, hasil tuning)."""

import csv
from pathlib import Path
from typing import Any


def _coerce(value: str) -> Any:
    """Ubah teks CSV menjadi int/float bila memungkinkan; sel kosong menjadi None."""
    if value == "":
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    """Baca CSV menjadi list of dict. Mengembalikan list kosong jika file tidak ada.

    Kolom indeks tanpa nama (hasil DataFrame.to_csv) diberi nama 'name'.
    """
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {(key or "name"): _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
