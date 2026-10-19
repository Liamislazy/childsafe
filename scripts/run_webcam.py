"""Live demo on one webcam. Video is NEVER written to disk; only an event CSV
(and, if asked, a per-frame feature CSV of numbers, used for B1 / tuning).

    python scripts/run_webcam.py --ckpt outputs/runs/efficientnet_b0_group/best.pt
    python scripts/run_webcam.py --no-classifier                     # Module B only
    python scripts/run_webcam.py --log-features --condition cam80cm_yaw30   # B1 recording session

Keys: q = quit, space = pause.
"""
import argparse
import math
import os
import time
from pathlib import Path

import cv2
import pandas as pd

from childsafe.alerts.event_log import EventLogger
from childsafe.alerts.processor import LEVEL_NAMES, Components
from childsafe.config import FULL_DATASET_CHECKPOINT, load_config
from childsafe.pipeline import FramePipeline
from childsafe.runtime.source import VideoSource
from childsafe.runtime import configure_runtime

COLORS = {0: (60, 180, 60), 1: (0, 190, 255), 2: (0, 0, 255)}   # BGR


def draw(frame, lm, out, fps):
    h, w = frame.shape[:2]
    for person in out.get('persons', []):
        pid = person['person_id']
        if pid in lm.boxes:
            x0,y0,x1,y1 = map(int, lm.boxes[pid])
            color = COLORS[person['level']] if person['level'] or person['green'] else (100,100,100)
            cv2.rectangle(frame, (x0,y0),(x1,y1),color,2)
            cv2.putText(frame, f"ID {pid}: {LEVEL_NAMES[person['level']]} / {person['observation']}",
                        (x0,max(60,y0)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    for obj in lm.objects:
        x0,y0,x1,y1 = map(int,obj.box)
        cv2.rectangle(frame,(x0,y0),(x1,y1),(255,180,0),1)
        cv2.putText(frame,obj.label,(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,180,0),1)
    lvl = out["level"]
    color = COLORS[lvl] if lvl or out.get('green') else (100, 100, 100)
    cv2.rectangle(frame, (0, 0), (w, 44), color, -1)
    calibration = ' [UNCALIBRATED]' if not out.get('calibrated', False) else ''
    txt = f"{LEVEL_NAMES[lvl].upper()} / {out.get('observation', 'degraded')}{calibration}"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    d = out["d"]
    info = (f"d={d:.2f} " if not math.isnan(d) else "d=-- ") + \
           f"near={int(out['near'])} dwell={out['dwell']:.1f}s obj={out['obj_status']}" + \
           (f"({out['obj_class']} p={out['p_unsafe']:.2f})" if not math.isnan(out['p_unsafe']) else "") + \
           f" fps={fps:.0f}"
    cv2.putText(frame, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    reasons = out.get('observation_reasons',[]) or [out.get('reason','')]
    cv2.putText(frame,'; '.join(reasons)[:120],(10, h-35),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(FULL_DATASET_CHECKPOINT),
                    help="local checkpoint; defaults to the imported full-data Kaggle model")
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--accurate", action="store_true",
                    help="GPU: use the larger yolo11m-seg detector (best for far objects; ~130-180 ms per frame)")
    ap.add_argument("--single-person", action="store_true",
                    help="follow only the largest person (ignore photos/screens/small faces in view)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", type=int)
    ap.add_argument('--source-env', help='environment variable holding video URL; do not put credentials in CLI/logs')
    ap.add_argument("--off", nargs="*", default=[], help="components to switch off (ablation)")
    ap.add_argument("--log-features", action="store_true")
    ap.add_argument("--condition", default="", help="label stored with feature logs (B1 sessions)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_runtime(cfg)
    if args.single_person:
        cfg['runtime']['primary_person_only'] = True
    if args.accurate:
        cfg['detector']['weights_gpu'] = str(Path(cfg['detector']['weights']).with_name('yolo11m-seg.pt'))
    rc = cfg["runtime"]
    ckpt = None if args.no_classifier else args.ckpt
    if ckpt is not None and not Path(ckpt).is_file():
        raise SystemExit("checkpoint not found; extract full_dataset_results.zip into outputs/full_dataset_results or pass --ckpt <best.pt>")
    pipe = FramePipeline(cfg, ckpt, Components.from_off(args.off))
    source_value = os.environ.get(args.source_env) if args.source_env else (
        args.camera if args.camera is not None else rc['camera_index'])
    if source_value is None:
        raise SystemExit('source environment variable missing')
    cap = VideoSource(source_value,rc['width'],rc['height']).start()
    logger = EventLogger(rc["log_dir"])
    feats = []
    t0 = time.perf_counter()
    fps, last, paused, t = 0.0, t0, False, 0.0
    session_id = None
    last_frame = None
    disconnected = False
    last_received_t = t0
    print('UNCALIBRATED elevated-view thresholds' if not cfg['quality']['calibrated'] else 'calibrated config')
    try:
        while True:
            packet = cap.read(.1)
            now = time.perf_counter()
            if packet is None or not packet.source_ok:
                if packet is None and now-last_received_t <= rc['stale_frame_s']:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                # A stalled read must not leave a green/danger display frozen indefinitely.
                if last_frame is not None:
                    t = now-t0
                    lm,row,out = pipe.process(cv2.cvtColor(last_frame,cv2.COLOR_BGR2RGB),t,
                                              frame_valid=False,source_ok=packet is None)
                    logger.update(out)
                    if args.log_features:
                        feats.extend({**f,'condition':args.condition} for f in out['feature_rows'])
                    cv2.imshow('childsafe',draw(last_frame.copy(),lm,out,fps))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                disconnected = True
                continue
            if session_id != packet.session_id or disconnected:
                if last_frame is not None:
                    _,_,ended = pipe.process(cv2.cvtColor(last_frame,cv2.COLOR_BGR2RGB),now-t0,
                                             frame_valid=False,source_ok=False)
                    logger.update(ended)
                    if args.log_features:
                        feats.extend({**f,'condition':args.condition} for f in ended['feature_rows'])
                pipe.reset()
                session_id = packet.session_id
                disconnected = False
            frame = packet.frame
            last_received_t = packet.received_t
            last_frame = frame.copy()
            t = packet.received_t-t0
            age = max(0.,now-packet.received_t)
            lm, row, out = pipe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t,
                                        received_t=packet.received_t,frame_age_s=age,
                                        processing_deadline=packet.received_t+rc['stale_frame_s'])
            logger.update(out)
            for row in out['feature_rows']:
                row['capture_slot_overwrites'] = cap.slot.overwrites
            if args.log_features:
                feats.extend({**f, 'condition': args.condition} for f in out['feature_rows'])
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now
            if out['inference_s'] > cfg['temporal']['max_gap_s']:
                print(f"processing too slow: inference {out['inference_s']:.3f}s; temporal continuity limited")
            if not paused:
                cv2.imshow("childsafe", draw(frame, lm, out, fps))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused
    finally:
        logger.close(t)
        cap.close()
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
