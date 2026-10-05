"""Evaluate a trained checkpoint in domain, out of domain (A2) and under corruptions (A5).

    python -m childsafe.classifier.evaluate --run outputs/runs/efficientnet_b0_group --ood data/ood
    python -m childsafe.classifier.evaluate --run outputs/runs/efficientnet_b0_group --corruptions

The OOD folder uses the same class folder names as ChildSUn (e.g. data/ood/knife/*.jpg);
extra sub-folders such as data/ood/low_light/knife/... are fine and reported by `condition`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T

from childsafe.classifier.engine import run_inference, summarize
from childsafe.classifier.models import load_checkpoint
from childsafe.config import load_config, normalize_name
from childsafe.data.catalog import build_catalog, source_roots
from childsafe.data.corruptions import KINDS, FixedCorruption
from childsafe.data.dataset import MEAN, STD, eval_transform, image_paths


_CACHE: dict = {}


def _cached_images(df, root, size):
    """Decode + resize + center-crop once; corruption passes reuse these (no DataLoader worker re-spawn,
    which costs tens of seconds per pass on Windows)."""
    key = (str(root), tuple(df["path"]), tuple(df["source"]) if "source" in df else (), size)
    if key not in _CACHE:
        tf = eval_transform(size).transforms[:2]
        _CACHE[key] = [tf[1](tf[0](Image.open(p).convert("RGB"))) for p in image_paths(df, root)]
    return _CACHE[key]


def _eval_df(model, ckpt, df, root, device, cfg, corruption=None, plot_prefix=None, title=""):
    size = ckpt["img_size"]
    imgs = _cached_images(df, root, size)
    tail = T.Compose(([FixedCorruption(*corruption)] if corruption and corruption[1] > 0 else [])
                     + [T.ToTensor(), T.Normalize(MEAN, STD)])
    idx = {c: i for i, c in enumerate(ckpt["classes"])}
    x = torch.stack([tail(im) for im in imgs])
    y = torch.tensor([idx[c] for c in df["category"]])
    lg, y = run_inference(model, DataLoader(TensorDataset(x, y), batch_size=64), device)
    rc = cfg["calibration"]
    return summarize(lg, y, ckpt["classes"], ckpt["binary_map"], ckpt["temperature"],
                     rc["reject_threshold"], rc["n_bins"], plot_prefix, title)


def _unknown_report(model, ckpt, df, root, device, cfg) -> dict:
    from childsafe.classifier.calibration import softmax
    size = ckpt["img_size"]
    imgs = _cached_images(df, root, size)
    tail = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    x = torch.stack([tail(im) for im in imgs])
    lg, _ = run_inference(model, DataLoader(TensorDataset(x, torch.zeros(len(x), dtype=torch.long)), batch_size=64), device)
    p = softmax(lg, ckpt["temperature"])
    unsafe = np.array([ckpt["binary_map"][c] == "unsafe" for c in ckpt["classes"]])
    pu = p[:, unsafe].sum(1)
    thr = cfg["calibration"]["reject_threshold"]
    status = np.where(np.maximum(pu, 1 - pu) < thr, "uncertain", np.where(pu >= 0.5, "unsafe", "safe"))
    pred = np.array(ckpt["classes"])[p.argmax(1)]
    out = {}
    for c in sorted(df["category"].unique()):
        m = (df["category"] == c).to_numpy()
        vc = pd.Series(pred[m]).value_counts(normalize=True).head(3).round(3).to_dict()
        out[c] = {"n": int(m.sum()), "unsafe": float((status[m] == "unsafe").mean()),
                  "uncertain": float((status[m] == "uncertain").mean()), "safe": float((status[m] == "safe").mean()),
                  "mean_max_prob": float(p[m].max(1).mean()), "top_predicted": vc}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder containing best.pt")
    ap.add_argument("--ood", default=None, help="out-of-domain image folder (A2)")
    ap.add_argument("--corruptions", action="store_true", help="A5 degradation curves on the test split")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run = Path(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(run / "best.pt", device)
    split_col = f"split_{ckpt.get('split', 'group')}"
    report: dict = {"run": run.name}

    if args.ood:
        ood = build_catalog(args.ood, cfg["label_map"], cfg.get("category_aliases"))
        unknown = set(ood["category"]) - set(ckpt["classes"])
        if unknown:
            # objects the model was never trained on (e.g. "other": pen, phone, hand): there is no correct
            # class, so report how the alert layer would treat them (unsafe = would raise danger near mouth)
            report["unknown_objects"] = _unknown_report(model, ckpt, ood[~ood["category"].isin(ckpt["classes"])],
                                                        args.ood, device, cfg)
            for c, r in report["unknown_objects"].items():
                print(f"[ood] '{c}' (not a trained class, n={r['n']}): called unsafe {r['unsafe']:.1%}, "
                      f"uncertain {r['uncertain']:.1%}, safe {r['safe']:.1%}; top predictions {r['top_predicted']}")
            ood = ood[ood["category"].isin(ckpt["classes"])]
        # condition = first folder level that is not the class name (e.g. low_light, dim, bright)
        ood["condition"] = [next((p for p in Path(x).parts[:-1] if normalize_name(p) not in cfg["label_map"]), "all")
                            for x in ood["path"]]
        report["ood"] = _eval_df(model, ckpt, ood, args.ood, device, cfg, plot_prefix=run / "ood", title="OOD")
        report["ood_by_condition"] = {c: _eval_df(model, ckpt, g, args.ood, device, cfg)
                                      for c, g in ood.groupby("condition")} if ood["condition"].nunique() > 1 else {}
        test = pd.read_csv(cfg["paths"]["splits"])
        test = test[test[split_col] == "test"]
        ind = _eval_df(model, ckpt, test, source_roots(cfg), device, cfg)
        report["in_domain_test"] = ind
        report["ood_drop"] = {"accuracy": ind["accuracy"] - report["ood"]["accuracy"],
                              "unsafe_recall": ind["binary"]["unsafe_recall"] - report["ood"]["binary"]["unsafe_recall"]}
        print(f"in-domain acc {ind['accuracy']:.4f} -> OOD acc {report['ood']['accuracy']:.4f}; "
              f"unsafe recall {ind['binary']['unsafe_recall']:.4f} -> {report['ood']['binary']['unsafe_recall']:.4f}; "
              f"ECE(cal) {ind['ece_calibrated']:.4f} -> {report['ood']['ece_calibrated']:.4f}")

    if args.corruptions:
        test = pd.read_csv(cfg["paths"]["splits"])
        test = test[test[split_col] == "test"]
        curves = {}
        for kind in KINDS:
            curves[kind] = []
            for sev in range(0, 6):
                m = _eval_df(model, ckpt, test, source_roots(cfg), device, cfg, (kind, sev))
                curves[kind].append({"severity": sev, "accuracy": m["accuracy"],
                                     "unsafe_recall": m["binary"]["unsafe_recall"], "ece": m["ece_calibrated"]})
                print(f"{kind} s{sev}: acc {m['accuracy']:.4f} unsafe_recall {m['binary']['unsafe_recall']:.4f}")
        report["corruption_curves"] = curves
        _plot_curves(curves, run / "degradation.png", run.name)

    (run / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {run / 'eval_report.json'}")


def _plot_curves(curves, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for kind, pts in curves.items():
        s = [p["severity"] for p in pts]
        axes[0].plot(s, [p["accuracy"] for p in pts], marker="o", label=kind)
        axes[1].plot(s, [p["unsafe_recall"] for p in pts], marker="o", label=kind)
    axes[0].set_title("accuracy"); axes[1].set_title("unsafe recall")
    for a in axes:
        a.set_xlabel("severity"); a.set_ylim(0, 1.02); a.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
