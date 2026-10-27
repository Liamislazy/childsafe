"""Copy verified exported model files into ignored Android assets; never download."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "outputs/android-models")
    args = p.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        p.error("Unsupported model manifest")
    files = []
    for role in ["classifier", "detector", "face", "hand"]:
        entry = manifest[role]
        path = args.source / entry["file"]
        if path.name != entry["file"] or not path.resolve().is_relative_to(args.source.resolve()):
            p.error("Model path escapes source directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            p.error(f"Model checksum mismatch: {role}")
        files.append(path)
        verification = entry.get("verification", {})
        if "input_file" in verification:
            vector = args.source / verification["input_file"]
            if vector.name != verification["input_file"] or not vector.resolve().is_relative_to(args.source.resolve()):
                p.error("Verification path escapes source directory")
            if hashlib.sha256(vector.read_bytes()).hexdigest() != verification["input_sha256"]:
                p.error("Verification tensor checksum mismatch")
            files.append(vector)
    destination = ROOT / "android/app/src/main/assets/models"
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copyfile(path, destination / path.name)
    shutil.copyfile(args.source / "manifest.json", destination / "manifest.json")
    print("Packaged four verified local models. Android camera behavior is not yet validated.")


if __name__ == "__main__":
    main()
