"""Smoke test untuk API yang sedang berjalan. Dipakai oleh CI/CD dan bisa dijalankan manual.

Pemakaian:  python -m scripts.smoke_test --base-url http://127.0.0.1:8000 --wait 60
"""

import argparse
import sys
import time
from io import BytesIO

import requests
from PIL import Image


def wait_until_healthy(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    sys.exit(f"GAGAL: API tidak sehat dalam {timeout:.0f} detik.")


def check(condition: bool, message: str) -> None:
    print(("OK    " if condition else "GAGAL ") + message)
    if not condition:
        sys.exit(1)


def sample_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (256, 256), (120, 90, 70)).save(buffer, format="JPEG")
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait", type=float, default=60, help="batas waktu menunggu API (detik)")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    wait_until_healthy(base, args.wait)
    health = requests.get(f"{base}/health", timeout=10).json()
    check(health["model_loaded"], f"/health: model dimuat ({health['model_name']})")

    info = requests.get(f"{base}/model-info", timeout=10).json()
    check(0 < info["threshold"] < 1, f"/model-info: threshold {info['threshold']}")

    files = {"file": ("smoke.jpg", sample_jpeg(), "image/jpeg")}
    response = requests.post(f"{base}/predict", files=files, params={"explain": "true"}, timeout=60)
    check(response.status_code == 200, f"/predict: status {response.status_code}")
    result = response.json()
    check(
        result["label"] in {"real", "fake"} and 0 <= result["p_fake"] <= 1,
        f"/predict: {result['label']} ({result['p_fake']:.3f})",
    )

    summary = requests.get(f"{base}/predictions/summary", timeout=10).json()
    check(
        summary["total"] >= 1,
        f"/predictions/summary: total {summary['total']} (tersimpan di database)",
    )
    print("Smoke test lulus.")


if __name__ == "__main__":
    main()
