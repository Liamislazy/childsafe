"""Run the existing local webcam pipeline and publish encrypted warning events.

Pairing and credentials are read from files, never command-line secrets.
No video/frames are stored or sent to the relay. Ctrl+C stops the camera.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from childsafe.companion.client import EventPublisher, read_pairing
from childsafe.companion.forwarder import CameraForwarder
from childsafe.config import ROOT, FULL_DATASET_CHECKPOINT, load_config
from childsafe.runtime import configure_runtime
from childsafe.runtime.source import VideoSource
from childsafe.web.app import RuntimeDemo


def main():
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument("--pairing-file", type=Path, help="One-use QR text/JSON from the monitor")
    p.add_argument("--credentials", type=Path, default=ROOT / "outputs/companion/webcam-credentials.json")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--cpu-threads", type=int)
    p.add_argument("--preview", action="store_true", help="Show a local preview; Q stops")
    args = p.parse_args()
    if args.credentials.exists():
        if args.pairing_file:
            p.error("Credentials already exist. Choose a different output file to pair another camera")
        credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
    else:
        if not args.pairing_file:
            p.error("Use --pairing-file once to pair this webcam")
        credentials = EventPublisher.pair(read_pairing(args.pairing_file))
        args.credentials.parent.mkdir(parents=True, exist_ok=True)
        with args.credentials.open("x", encoding="utf-8") as out:
            json.dump(credentials, out)
    checkpoint = args.checkpoint or next((path for path in [FULL_DATASET_CHECKPOINT,
        ROOT / "outputs/with_other/runs/efficientnet_b0_group_robust/best.pt"] if path.is_file()), None)
    if checkpoint is None or not checkpoint.is_file():
        p.error("Local classifier checkpoint missing; supply --checkpoint")
    cfg = load_config(str(args.config) if args.config else None)
    if args.cpu_threads is not None:
        cfg["runtime"]["cpu_threads"] = args.cpu_threads
    configure_runtime(cfg)
    runtime = RuntimeDemo(cfg, {"companion webcam": str(checkpoint)})
    publisher = EventPublisher(credentials, clock=time.perf_counter).start()
    forwarder = CameraForwarder(publisher, frame_max_age=cfg["runtime"].get("stale_frame_s", .25))
    source = VideoSource(args.camera_index, width=cfg["runtime"]["width"], height=cfg["runtime"]["height"]).start()
    session = None
    print("Webcam running locally. Event-only encrypted transmission. Ctrl+C stops.")
    try:
        while True:
            packet = source.read()
            if packet is None or not packet.source_ok:
                forwarder.unavailable("Không nhận được ảnh mới từ webcam")
                continue
            if session != packet.session_id:
                runtime.reset()
                forwarder.reset()
                session = packet.session_id
            age = time.perf_counter()-packet.received_t
            if age > cfg["runtime"].get("stale_frame_s", .25):
                forwarder.unavailable("Ảnh webcam đã cũ trước khi xử lý")
                continue
            ok, encoded = cv2.imencode(".jpg", packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                forwarder.unavailable("Không đọc được ảnh webcam")
                continue
            try:
                result = runtime.frame(encoded.tobytes(), "", 0, "hand", frame_age_s=age,
                                       deadline=packet.received_t+cfg["runtime"].get("stale_frame_s", .25))
                if not forwarder.result(result, packet.received_t):
                    runtime.reset()
            except Exception:
                runtime.reset()
                forwarder.unavailable("Runtime không phân tích được ảnh hiện tại")
                continue
            # This service does not collect a research recording in RAM.
            with runtime.lock:
                runtime.rows.clear()
            if args.preview:
                cv2.imshow("ChildSafe local webcam - Q to stop", packet.frame)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        publisher.close()
        runtime.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
