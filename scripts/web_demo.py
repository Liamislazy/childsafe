"""Local multi-person alert runtime, with a separate photo classifier comparison.

    python scripts/web_demo.py
    python scripts/web_demo.py --model standard=path/to/best.pt --model robust=path/to/best.pt

The first selected checkpoint drives live alerts. Open http://127.0.0.1:8000 and
click "Start camera". Localhost only; frames are never saved. Numeric session
features remain in memory until reset and can be downloaded explicitly.
"""
import argparse
from pathlib import Path

import uvicorn

from childsafe.config import ROOT, FULL_DATASET_CHECKPOINT, load_config
from childsafe.web.app import RuntimeDemo, create_app

RUNS = ROOT / "outputs" / "kaggle_v3" / "childsafe" / "outputs" / "runs"
DEFAULT = {"full dataset (Kaggle)": FULL_DATASET_CHECKPOINT,
           "standard": RUNS / "efficientnet_b0_group" / "best.pt",
           "robust (A5)": RUNS / "efficientnet_b0_group_robust" / "best.pt",
           "robust + other (Open Images)": ROOT / "outputs" / "with_other" / "runs" / "efficientnet_b0_group_robust" / "best.pt",
           "robust + other (102 OI classes)": ROOT / "outputs" / "with_other_train" / "runs" / "efficientnet_b0_group_robust" / "best.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path/to/best.pt (repeatable)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model) if args.model else {k: str(v) for k, v in DEFAULT.items() if v.exists()}
    if not models:
        raise SystemExit("No local checkpoint found. Extract full_dataset_results.zip into outputs/full_dataset_results or pass --model name=path/to/best.pt")
    for name, p in models.items():
        if not Path(p).exists():
            raise SystemExit(f"checkpoint for '{name}' not found: {p}")
    demo = RuntimeDemo(load_config(args.config), models)
    print(f"open http://127.0.0.1:{args.port}")
    try:
        uvicorn.run(create_app(demo), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
