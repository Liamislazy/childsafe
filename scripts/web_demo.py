"""Local multi-person live-camera alert runtime.

    python scripts/web_demo.py
    python scripts/web_demo.py --model standard=path/to/best.pt --model robust=path/to/best.pt

The first existing checkpoint drives live alerts (only that one is loaded). Open http://127.0.0.1:8000 and
click "Start camera". Localhost only; frames are never saved. Numeric session
features remain in memory until reset and can be downloaded explicitly.
"""
import argparse
from pathlib import Path

import uvicorn

from childsafe.config import ROOT, FULL_DATASET_CHECKPOINT, MOBILENET_CHECKPOINT, load_config
from childsafe.web.app import RuntimeDemo, create_app
from childsafe.runtime import configure_runtime

RUNS = ROOT / "outputs" / "kaggle_v3" / "childsafe" / "outputs" / "runs"
# The FIRST existing checkpoint drives live alerts. MobileNetV3-Large is the chosen live model; it has no
# "other" class, so unfamiliar objects (pens, phones) are forced into a ChildSUn class more often than with
# the "other" models below (docs/results_openimages.md). Those stay as fallbacks.
DEFAULT = {"MobileNetV3-Large": MOBILENET_CHECKPOINT,
           "full dataset (Kaggle)": FULL_DATASET_CHECKPOINT,
           "robust + other (Open Images)": ROOT / "outputs" / "with_other" / "runs" / "efficientnet_b0_group_robust" / "best.pt",
           "standard": RUNS / "efficientnet_b0_group" / "best.pt",
           "robust (A5)": RUNS / "efficientnet_b0_group_robust" / "best.pt"}
# Not shown by default; add with --model "robust + other (102 OI classes)=outputs/with_other_train/runs/efficientnet_b0_group_robust/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path/to/best.pt (repeatable)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument('--cpu-threads', type=int, help='batch-one CPU thread budget; 0 restores library defaults')
    ap.add_argument('--object-model', help='local trained object-only .pt; keeps the primary person model')
    ap.add_argument("--accurate", action="store_true",
                    help="GPU: use the larger yolo11m-seg detector (best for far objects; ~130-180 ms per frame)")
    ap.add_argument("--single-person", action="store_true",
                    help="follow only the largest person (ignore photos/screens/small faces in view)")
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model) if args.model else {k: str(v) for k, v in DEFAULT.items() if v.exists()}
    if not models:
        raise SystemExit(f"No local checkpoint found. Expected {MOBILENET_CHECKPOINT} or pass --model name=path/to/best.pt")
    for name, p in models.items():
        if not Path(p).exists():
            raise SystemExit(f"checkpoint for '{name}' not found: {p}")
    cfg = load_config(args.config)
    if args.cpu_threads is not None:
        cfg['runtime']['cpu_threads'] = args.cpu_threads
    if args.object_model:
        if not Path(args.object_model).is_file():
            raise SystemExit('Supplemental object checkpoint does not exist; train/download it explicitly first.')
        cfg['detector']['supplemental_weights'] = str(Path(args.object_model).resolve())
    configure_runtime(cfg)
    if args.single_person:
        cfg['runtime']['primary_person_only'] = True
    if args.accurate:
        cfg['detector']['weights_gpu'] = str(Path(cfg['detector']['weights']).with_name('yolo11m-seg.pt'))
    demo = RuntimeDemo(cfg, models)
    print(f"open http://127.0.0.1:{args.port}")
    try:
        uvicorn.run(create_app(demo), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
