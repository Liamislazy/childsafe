"""Download ChildSUn from Kaggle and save its metadata (license etc.).

Needs Kaggle credentials: either a new-style token (KGAT_...) in the KAGGLE_API_TOKEN environment
variable or in %USERPROFILE%\\.kaggle\\access_token, or a legacy %USERPROFILE%\\.kaggle\\kaggle.json.
Manual alternative: download the zip from
https://www.kaggle.com/datasets/fairuzkhan002/datasetvgg19 and extract it into data/raw/childsun.

    python scripts/download_childsun.py
"""
import json
import subprocess
import sys
from pathlib import Path

from childsafe.config import load_config

SLUG = "fairuzkhan002/datasetvgg19"


def main():
    cfg = load_config()
    dest = Path(cfg["paths"]["raw_data"])
    dest.mkdir(parents=True, exist_ok=True)
    kaggle = [sys.executable, "-m", "kaggle"]
    subprocess.run(kaggle + ["datasets", "metadata", SLUG, "-p", str(dest)], check=True)
    subprocess.run(kaggle + ["datasets", "download", SLUG, "-p", str(dest), "--unzip"], check=True)
    meta_file = dest / "dataset-metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        print("title:", meta.get("title"))
        print("licenses:", meta.get("licenses"))
    print(f"downloaded to {dest}. Next: python scripts/inspect_dataset.py")


if __name__ == "__main__":
    main()
