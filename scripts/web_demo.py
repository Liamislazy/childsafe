"""Local web demo comparing two classifiers side by side (default: standard vs A5-robust EfficientNet-B0).

    python scripts/web_demo.py
    python scripts/web_demo.py --model standard=path/to/best.pt --model robust=path/to/best.pt

Then open http://127.0.0.1:8000 and click "Start camera". Listens on localhost only; nothing is saved.
"""
import argparse
from pathlib import Path

import uvicorn

from childsafe.config import ROOT, load_config
from childsafe.web.app import Demo, create_app

RUNS = ROOT / "outputs" / "kaggle_v3" / "childsafe" / "outputs" / "runs"
DEFAULT = {"standard": RUNS / "efficientnet_b0_group" / "best.pt",
           "robust (A5)": RUNS / "efficientnet_b0_group_robust" / "best.pt",
           "robust + other (Open Images)": ROOT / "outputs" / "with_other" / "runs" / "efficientnet_b0_group_robust" / "best.pt",
           "robust + other (102 OI classes)": ROOT / "outputs" / "with_other_train" / "runs" / "efficientnet_b0_group_robust" / "best.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path/to/best.pt (repeatable)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model) if args.model else         {k: str(v) for k, v in DEFAULT.items() if Path(v).exists()}
    for name, p in models.items():
        if not Path(p).exists():
            raise SystemExit(f"checkpoint for '{name}' not found: {p}")
    demo = Demo(load_config(args.config), models)
    print(f"open http://127.0.0.1:{args.port}")
    uvicorn.run(create_app(demo), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
