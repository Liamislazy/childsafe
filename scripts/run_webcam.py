"""Live demo on one webcam. Video is NEVER written to disk; only an event CSV
(and, if asked, a per-frame feature CSV of numbers, used for B1 / tuning).

    python scripts/run_webcam.py --ckpt outputs/runs/efficientnet_b0_group/best.pt
    python scripts/run_webcam.py --no-classifier                     # Module B only
    python scripts/run_webcam.py --log-features --condition cam80cm_yaw30   # B1 recording session

Keys: q = quit, space = pause.
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import pandas as pd

from childsafe.alerts.event_log import EventLogger
from childsafe.alerts.processor import LEVEL_NAMES, Components
from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.vision.geometry import MOUTH

COLORS = {0: (60, 180, 60), 1: (0, 190, 255), 2: (0, 0, 255)}   # BGR


def draw(frame, lm, out, fps):
    h, w = frame.shape[:2]
    if lm.face is not None:
        m = lm.face[MOUTH, :2].mean(0).astype(int)
        cv2.circle(frame, tuple(m), 6, (255, 0, 255), -1)
    for hand in lm.hands:
        for p in hand[:, :2].astype(int):
            cv2.circle(frame, tuple(p), 2, (255, 255, 0), -1)
    lvl = out["level"]
    color = COLORS[lvl] if lvl or out.get('green') else (100, 100, 100)
    cv2.rectangle(frame, (0, 0), (w, 44), color, -1)
    txt = f"{LEVEL_NAMES[lvl].upper()} / {out.get('observation', 'degraded')}  {out['reason']}"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    d = out["d"]
    info = (f"d={d:.2f} " if not math.isnan(d) else "d=-- ") + \
           f"near={int(out['near'])} dwell={out['dwell']:.1f}s obj={out['obj_status']}" + \
           (f"({out['obj_class']} p={out['p_unsafe']:.2f})" if not math.isnan(out['p_unsafe']) else "") + \
           f" fps={fps:.0f}"
    cv2.putText(frame, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", type=int)
    ap.add_argument("--off", nargs="*", default=[], help="components to switch off (ablation)")
    ap.add_argument("--log-features", action="store_true")
    ap.add_argument("--condition", default="", help="label stored with feature logs (B1 sessions)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rc = cfg["runtime"]
    ckpt = None if args.no_classifier else args.ckpt
    if ckpt is None and not args.no_classifier:
        raise SystemExit("pass --ckpt <best.pt> or --no-classifier")
    pipe = FramePipeline(cfg, ckpt, Components.from_off(args.off))
    cap = cv2.VideoCapture(args.camera if args.camera is not None else rc["camera_index"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rc["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rc["height"])
    if not cap.isOpened():
        raise SystemExit("cannot open camera")
    logger = EventLogger(rc["log_dir"])
    feats = []
    t0 = time.perf_counter()
    fps, last, paused, t = 0.0, t0, False, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t = time.perf_counter() - t0
            lm, row, out = pipe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t)
            logger.update(out)
            if args.log_features:
                feats.append({**row, "condition": args.condition, "level": out["level"]})
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now
            if not paused:
                cv2.imshow("childsafe", draw(frame, lm, out, fps))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused
    finally:
        logger.close(t)
        cap.release()
        cv2.destroyAllWindows()
        pipe.close()
        print(f"events: {logger.path}")
        if feats:
            p = Path(cfg["paths"]["outputs"]) / "features" / f"live_{args.condition or 'session'}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(feats).to_csv(p, index=False)
            print(f"features: {p}")


if __name__ == "__main__":
    main()
