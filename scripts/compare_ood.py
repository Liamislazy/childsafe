"""Compare checkpoints on (1) the ChildSUn test split and (2) Open Images *test* crops (never trained on),
including how often objects outside the ChildSUn classes ("other": pen, phone, hand...) are called unsafe.

    python scripts/compare_ood.py --model name=path/to/best.pt --model name2=path2/best.pt
Writes docs/results_openimages.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from childsafe.classifier.calibration import softmax
from childsafe.classifier.models import load_checkpoint
from childsafe.config import ROOT, load_config
from childsafe.data.dataset import eval_transform

OI = ROOT / "data" / "external" / "openimages"


def predict(model, ckpt, paths, device):
    tf = eval_transform(ckpt['img_size'], preprocessing=ckpt.get('preprocessing','center_crop'))
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 64]]).to(device)
            out.append(model(x).float().cpu().numpy())
    p = softmax(np.concatenate(out), ckpt["temperature"])
    classes = np.array(ckpt["classes"])
    unsafe = np.array([ckpt["binary_map"][c] == "unsafe" for c in classes])
    unknown = np.array([ckpt['binary_map'][c] == 'unknown' or c in ('other','unknown','empty_hand') for c in classes])
    return classes[p.argmax(1)], p[:, unsafe].sum(1), p.max(1), p[:,unknown].sum(1)


def status(pu, thr, class_conf, predicted, unknown_mass):
    rejected = ((np.maximum(pu,1-pu) < thr) | (class_conf < thr) |
                np.isin(predicted,['other','unknown','empty_hand']) | (unknown_mass >= .5))
    return np.where(rejected,'uncertain',np.where(pu >= .5,'unsafe','safe'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path/to/best.pt")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    thr = cfg["calibration"]["reject_threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = pd.read_csv(cfg["paths"]["splits"])
    cs = sp[sp.split_group == "test"]
    cs_paths = [Path(cfg["paths"]["raw_data"]) / p for p in cs.path]
    man = pd.read_csv(OI / "manifest.csv")
    oi = man[(man.split == "test") & man.status.isin(["ok", "exists"])].reset_index(drop=True)
    oi_paths = [OI / p for p in oi.path]
    unsafe_cls = {c for c, b in cfg["label_map"].items() if b == "unsafe"}

    rows, other_rows = [], []
    for spec in args.model:
        name, path = spec.split("=", 1)
        model, ck = load_checkpoint(path, device)
        pred, pu, _, _ = predict(model, ck, cs_paths, device)
        y_uns = cs.label.eq("unsafe").to_numpy()
        r = {"model": name, "classes": len(ck["classes"]),
             "ChildSUn acc": (pred == cs.category.to_numpy()).mean(),
             "ChildSUn unsafe recall": (pu[y_uns] >= 0.5).mean()}
        pred, pu, conf, unknown = predict(model, ck, oi_paths, device)
        st = status(pu,thr,conf,pred,unknown)
        known = oi.category.ne("other").to_numpy()
        y = oi.category.to_numpy()
        y_uns = oi.category.isin(unsafe_cls).to_numpy()
        r["Open Images acc (9 shared classes)"] = (pred[known] == y[known]).mean()
        r["Open Images unsafe recall"] = (pu[y_uns] >= 0.5).mean()
        oth = ~known
        r["other: called unsafe"] = (st[oth] == "unsafe").mean()
        r["other: predicted 'other'"] = (pred[oth] == "other").mean() if "other" in ck["classes"] else float("nan")
        pen = (oi.oi_class == "Pen").to_numpy()
        r["pens: called unsafe"] = (st[pen] == "unsafe").mean()
        rows.append(r)
        d = pd.DataFrame({"object": oi.oi_class[oth].to_numpy(), "unsafe": st[oth] == "unsafe"})
        other_rows.append(d.groupby("object").unsafe.mean().rename(name))
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    t = pd.DataFrame(rows).set_index("model").T
    o = pd.concat(other_rows, axis=1)
    o.insert(0, "n", oi[oi.category == "other"].oi_class.value_counts())
    def fmt(v):
        if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer() and v > 1):
            return str(int(v))
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.1%}"
        return str(v)
    def md(df):
        head = "| | " + " | ".join(map(str, df.columns)) + " |\n|---|" + "---|" * len(df.columns)
        return head + "\n" + "\n".join(f"| {i} | " + " | ".join(fmt(v) for v in r) + " |" for i, r in df.iterrows())
    text = (f"# Out-of-domain test on Open Images V7 crops\n\n"
            f"Open Images *test* split crops (never used for training): {len(oi)} crops, "
            f"{int((oi.category != 'other').sum())} of the ChildSUn classes and {int((oi.category == 'other').sum())} 'other' objects. "
            f"ChildSUn test = canonical group split (checksum 42e3fb2547c5).\n"
            f"Shared classes in Open Images: few scissors ({int((oi.category == 'scissors').sum())}), heaters "
            f"({int((oi.category == 'heater').sum())}) and sockets ({int((oi.category == 'socket').sum())}); no baby items.\n\n"
            f"## Summary\n\n{md(t)}\n\n'called unsafe' = calibrated P(unsafe) >= 0.5, category and binary confidence >= {thr}, excluding unknown/other predictions. "
            f"(below that it is 'uncertain', which the alert layer treats as caution).\n\n"
            f"## Share of 'other' objects called unsafe, by object\n\n{md(o)}\n")
    out = Path(cfg["paths"]["docs"]) / "results_openimages.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
