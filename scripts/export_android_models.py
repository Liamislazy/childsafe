"""Export local classifier/YOLO weights and verify classifier logit parity.

No downloads or training. Export is an intermediate artifact, not validation
of Android camera accuracy. Preprocessing/calibration metadata is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from childsafe.classifier.models import load_checkpoint
from childsafe.data.dataset import eval_transform
from childsafe.config import load_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_classifier(checkpoint: Path, output: Path):
    import onnxruntime as ort
    from PIL import Image

    torch.set_num_threads(1)
    model, metadata = load_checkpoint(str(checkpoint), "cpu")
    size = int(metadata["img_size"])
    preprocessing = metadata.get("preprocessing", "center_crop")
    if preprocessing not in {"center_crop", "letterbox"}:
        raise ValueError("Unsupported checkpoint preprocessing")
    transform = eval_transform(size, preprocessing=preprocessing)
    generator = np.random.default_rng(5678)
    samples = [transform(Image.fromarray(generator.integers(0, 256, (h, w, 3), dtype=np.uint8))).unsqueeze(0)
               for h, w in [(size, size), (size + 43, size - 17), (size - 37, size + 61)]]
    target = output / "classifier.onnx"
    torch.onnx.export(model, samples[0], str(target), input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(target), sess_options=options, providers=["CPUExecutionProvider"])
    errors = []
    for sample in samples:
        with torch.inference_mode():
            reference = model(sample).numpy()
        actual = session.run(["logits"], {"image": sample.numpy()})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
        errors.append(float(np.max(np.abs(actual - reference))))
    return {"file": target.name, "sha256": sha256(target), "source_sha256": sha256(checkpoint),
            "arch": metadata["arch"], "classes": metadata["classes"],
            "binary_map": metadata["binary_map"], "img_size": size,
            "preprocessing": preprocessing, "temperature": float(metadata.get("temperature", 1.0)),
            "unknown_classes": [name for name in metadata["classes"] if metadata["binary_map"][name] == "unknown" or name in {"other", "unknown", "empty_hand"}],
            "resize_short_edge": int(size * 1.14) if preprocessing == "center_crop" else None,
            "letterbox_fill": [124, 116, 104], "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "layout": "NCHW", "output": "logits", "opset": 17,
            "verification": {"kind": "three synthetic inputs, CPU logits only", "max_abs_errors": errors}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--detector", type=Path)
    p.add_argument("--face", type=Path)
    p.add_argument("--hand", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/android-models"))
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    for path in [args.checkpoint, args.detector, args.face, args.hand]:
        if path is not None and not path.is_file():
            p.error(f"Local model missing: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "mobile_camera_validated": False,
                "classifier": export_classifier(args.checkpoint, args.out)}
    cfg = load_config(str(args.config) if args.config else None)
    manifest["runtime_config"] = {name: cfg[name] for name in ["calibration", "landmarks", "smoothing", "temporal", "rules", "quality", "hazard_policy"]}
    if args.detector:
        from ultralytics import YOLO
        model = YOLO(str(args.detector))
        if model.task != "segment":
            p.error("Mobile runtime requires a segmentation detector; boxes alone cannot confirm object proximity")
        source = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=False,
                                   dynamic=False, nms=False, device="cpu"))
        target = args.out / "detector.onnx"
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        manifest["detector"] = {"file": target.name, "sha256": sha256(target),
                                "source_sha256": sha256(args.detector), "classes": model.names,
                                "input_size": 640, "task": "segment", "mask_coefficients": 32,
                                "output": "raw YOLO11 segmentation, no NMS", "opset": 17}
    for name, path in [("face", args.face), ("hand", args.hand)]:
        if path:
            target = args.out / f"{name}_landmarker.task"
            shutil.copyfile(path, target)
            manifest[name] = {"file": target.name, "sha256": sha256(target), "source_sha256": sha256(path)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
