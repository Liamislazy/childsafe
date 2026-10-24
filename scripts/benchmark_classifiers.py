"""Benchmark Module A classifiers trained on the canonical group split (checksum 42e3fb2547c5).

For every checkpoint: test metrics from its run's metrics.json (written by childsafe.classifier.train), accuracy
under the same fixed-seed corruptions (severity 3 and 5), Open Images test crops (never trained on), parameters,
checkpoint size and single-image latency on this machine (GPU, CPU all threads, CPU 1 thread).

    python scripts/benchmark_classifiers.py --model "EfficientNet-B0=outputs/.../best.pt" --model ... \
        --out outputs/benchmark/benchmark.json

Nothing here selects a model or threshold: the test split is only reported.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_ood  # noqa: E402  (predict/status: same OOD rules as docs/results_openimages.md)

from childsafe.classifier.engine import run_inference, summarize  # noqa: E402
from childsafe.classifier.evaluate import _cached_images  # noqa: E402
from childsafe.classifier.models import load_checkpoint  # noqa: E402
from childsafe.config import load_config  # noqa: E402
from childsafe.data.catalog import source_roots  # noqa: E402
from childsafe.data.corruptions import KINDS, FixedCorruption  # noqa: E402
from childsafe.data.dataset import MEAN, STD  # noqa: E402

SEVERITIES = (3, 5)


def latency_ms(model, device, runs, warmup, threads=None):
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    model = model.to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    times = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(old)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path/to/best.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rc = cfg["calibration"]
    gpu = torch.device("cuda") if torch.cuda.is_available() else None

    sp = pd.read_csv(cfg["paths"]["splits"])
    test = sp[sp.split_group == "test"].reset_index(drop=True)
    imgs = _cached_images(test, source_roots(cfg), 224, "center_crop")
    sets = {("clean", 0): None}
    sets.update({(k, s): None for k in KINDS for s in SEVERITIES})
    for key in sets:   # one fixed-seed corruption pass shared by every model; uint8 keeps 13 sets in ~1.5 GB
        tf = T.Compose(([FixedCorruption(*key)] if key[1] else []) + [T.PILToTensor()])
        sets[key] = torch.stack([tf(im) for im in imgs])
    norm = T.Normalize(MEAN, STD)
    print(f"prepared {len(sets)} test sets of {len(imgs)} images", flush=True)

    oi = pd.read_csv(compare_ood.OI / "manifest.csv")
    oi = oi[(oi.split == "test") & oi.status.isin(["ok", "exists"])].reset_index(drop=True)
    oi_paths = [compare_ood.OI / p for p in oi.path]
    known = oi.category.ne("other").to_numpy()
    pen = (oi.oi_class == "Pen").to_numpy()
    unsafe_cls = {c for c, b in cfg["label_map"].items() if b == "unsafe"}
    oi_unsafe = oi.category.isin(unsafe_cls).to_numpy()

    results = {"machine": {"cpu": platform.processor(), "cpu_threads": torch.get_num_threads(),
                           "gpu": torch.cuda.get_device_name(0) if gpu else None, "torch": torch.__version__},
               "models": []}
    for spec in args.model:
        name, path = spec.split("=", 1)
        path = Path(path)
        model, ck = load_checkpoint(path, gpu or "cpu")
        assert ck["img_size"] == 224 and ck.get("preprocessing", "center_crop") == "center_crop", path
        m = json.loads((path.parent / "metrics.json").read_text())
        assert m["split"] == "group", path
        t = m["test"]
        r = {"name": name, "arch": ck["arch"], "checkpoint": str(path), "best_epoch": m["best_epoch"],
             "epochs_run": len(m["history"]),
             "train_sec_per_epoch": statistics.mean(h["sec"] for h in m["history"]),
             "accuracy": t["accuracy"], "macro_f1": t["macro_f1"],
             "unsafe_recall": t["binary"]["unsafe_recall"], "unsafe_precision": t["binary"]["unsafe_precision"],
             "binary_ece_cal": t["binary_ece_calibrated"], "ece_cal": t["ece_calibrated"]}

        y = torch.tensor([ck["classes"].index(c) for c in test.category])
        corr = {}
        for (kind, sev), x in sets.items():
            loader = ((norm(xb.float() / 255), yb) for xb, yb in DataLoader(TensorDataset(x, y), batch_size=64))
            lg, yy = run_inference(model, loader, gpu or torch.device("cpu"))
            s = summarize(lg, yy, ck["classes"], ck["binary_map"], ck["temperature"], rc["reject_threshold"], rc["n_bins"])
            corr[f"{kind}_s{sev}"] = {"accuracy": s["accuracy"], "unsafe_recall": s["binary"]["unsafe_recall"]}
        r["clean_check_accuracy"] = corr["clean_s0"]["accuracy"]   # must equal metrics.json accuracy
        r["corruptions"] = corr
        for sev in SEVERITIES:
            r[f"mean_corruption_acc_s{sev}"] = statistics.mean(corr[f"{k}_s{sev}"]["accuracy"] for k in KINDS)

        pred, pu, conf, unk = compare_ood.predict(model, ck, oi_paths, gpu or torch.device("cpu"))
        st = compare_ood.status(pu, rc["reject_threshold"], conf, pred, unk)
        r["oi_acc_shared"] = float((pred[known] == oi.category.to_numpy()[known]).mean())
        r["oi_unsafe_recall"] = float((pu[oi_unsafe] >= 0.5).mean())
        r["oi_other_called_unsafe"] = float((st[~known] == "unsafe").mean())
        r["oi_pens_called_unsafe"] = float((st[pen] == "unsafe").mean())

        r["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
        r["checkpoint_mb"] = path.stat().st_size / 1e6
        if gpu:
            r["latency_gpu_ms"] = latency_ms(model, gpu, runs=100, warmup=20)
        model = model.cpu()
        r["latency_cpu_ms"] = latency_ms(model, torch.device("cpu"), runs=50, warmup=10)
        r["latency_cpu_1thread_ms"] = latency_ms(model, torch.device("cpu"), runs=20, warmup=3, threads=1)
        results["models"].append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "corruptions"}),
              flush=True)
        del model
        if gpu:
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
