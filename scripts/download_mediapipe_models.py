"""Download the MediaPipe Face Landmarker and Hand Landmarker model bundles (official Google storage).

    python scripts/download_mediapipe_models.py
"""
import urllib.request
from pathlib import Path

from childsafe.config import load_config

URLS = {
    "face_model": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_model": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def main():
    cfg = load_config()
    for key, url in URLS.items():
        dest = Path(cfg["paths"][key])
        if dest.exists():
            print(f"exists: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
