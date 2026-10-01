"""Run Module B (+ Module A on every hand frame) over recorded scenario videos and save
per-frame feature CSVs. Replay/evaluation then works on these numbers only.

    python scripts/extract_features.py --videos data/recordings --ckpt outputs/runs/efficientnet_b0_group/best.pt

Recordings must come from consenting adult volunteers / dolls (docs/ethics.md). Delete them
at the end of the project; the feature CSVs contain no images.
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="video file or folder")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--classify-input", choices=["hand_crop", "frame"], default="hand_crop")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    out_dir = Path(args.out or Path(cfg["paths"]["outputs"]) / "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in videos:
        pipe = FramePipeline(cfg, args.ckpt, classify_all=True, classify_input=args.classify_input)
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows, i = [], 0
        for _ in tqdm(range(n), desc=v.name):
            ok, frame = cap.read()
            if not ok:
                break
            _, row, _ = pipe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), i / fps)
            rows.append({**row, "video": v.stem, "fps": fps})
            i += 1
        cap.release()
        pipe.close()
        dest = out_dir / f"{v.stem}.csv"
        pd.DataFrame(rows).to_csv(dest, index=False)
        print(f"{v.name}: {i} frames, face {pd.DataFrame(rows).face_ok.mean():.1%}, hand {pd.DataFrame(rows).hand_ok.mean():.1%} -> {dest}")


if __name__ == "__main__":
    main()
