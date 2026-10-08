"""Train one Module A model on one split, then calibrate on val and evaluate on test.

    python -m childsafe.classifier.train --arch efficientnet_b0 --split group
    python -m childsafe.classifier.train --arch efficientnet_b0 --split random   # leakage comparison
    python -m childsafe.classifier.train --arch efficientnet_b0 --split group --robust  # A5

Outputs go to outputs/runs/<arch>_<split>[_robust]/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from childsafe.classifier.calibration import fit_temperature
from childsafe.classifier.engine import run_inference, summarize
from childsafe.classifier.models import create_model, save_checkpoint
from childsafe.config import load_config
from childsafe.data.catalog import source_roots
from childsafe.data.dataset import CatalogDataset, eval_transform, train_transform


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def class_balanced_sampler(targets, num_classes, balance):
    if balance != 'sqrt':
        return None
    counts = torch.bincount(targets,minlength=num_classes).float().clamp(min=1)
    return torch.utils.data.WeightedRandomSampler(counts.rsqrt()[targets],len(targets),replacement=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--split", choices=["group", "random"], default="group")
    ap.add_argument("--robust", action="store_true", help="A5 robustness augmentation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", help="Reviewed expanded manifest with train/val/cal/test in split_group")
    ap.add_argument("--balanced", action="store_true", help="Inverse-square-root training class weights")
    ap.add_argument("--preprocessing", choices=["center_crop", "letterbox"], default="center_crop")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress visible when redirected to a log

    cfg = load_config(args.config)
    tc = cfg["train"]
    epochs = args.epochs or tc["epochs"]
    bs = args.batch_size or tc["batch_size"]
    seed = cfg["split"]["seed"] if args.seed is None else args.seed
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tc["amp"] and device.type == "cuda"

    df = pd.read_csv(args.manifest or cfg["paths"]["splits"])
    col = f"split_{args.split}"
    classes = sorted(df["category"].unique())
    binary_map = dict(df[["category", "label"]].drop_duplicates().values)
    if df.groupby("category")["label"].nunique().max() != 1:
        raise ValueError("Conflicting category risk labels")
    if args.manifest:
        if args.split != "group" or not set(df[col]).issubset({"train", "val", "cal", "test"}):
            raise ValueError("Expanded training requires group split with train/val/cal/test")
        if df.groupby("group")[col].nunique().max() != 1:
            raise ValueError("Source group leakage across partitions")
        for part in ("train", "val", "cal", "test"):
            missing = set(classes) - set(df.loc[df[col] == part, "category"])
            if missing:
                raise ValueError(f"{part} missing classes: {sorted(missing)}; collect more independent sources")
    root = source_roots(cfg)   # ChildSUn + any extra sources
    size = tc["img_size"]

    def loader(part, tf, shuffle):
        ds = CatalogDataset(df[df[col] == part], root, classes, tf)
        sampler = class_balanced_sampler(ds.targets,len(classes),tc.get('balance')) if shuffle else None
        return DataLoader(ds, batch_size=bs, shuffle=shuffle and sampler is None, sampler=sampler,
                          num_workers=tc["num_workers"], pin_memory=device.type == "cuda",
                          persistent_workers=tc["num_workers"] > 0, drop_last=shuffle and len(ds) > bs)

    eval_tf = eval_transform(size, preprocessing=args.preprocessing)
    train_dl = loader("train", train_transform(size, args.robust, args.preprocessing), True)
    val_dl = loader("val", eval_tf, False)
    cal_dl = loader("cal", eval_tf, False) if args.manifest else val_dl
    test_dl = loader("test", eval_tf, False)

    run_name = f"{args.arch}_{args.split}" + ("_robust" if args.robust else "")
    out = Path(args.out or Path(cfg["paths"]["outputs"]) / "runs" / run_name)
    out.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        df.to_csv(out / "training_manifest.csv", index=False)
    provenance = {"arguments": vars(args), "seed": seed, "config": cfg,
                  "manifest_sha256": hashlib.sha256(Path(args.manifest or cfg["paths"]["splits"]).read_bytes()).hexdigest(),
                  "partition_counts": df.groupby(col).size().to_dict()}
    (out / "run_config.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    lr = tc.get("lr_overrides", {}).get(args.arch, tc["lr"])
    print(f"[{run_name}] lr={lr} epochs={epochs} batch={bs}", flush=True)
    model = create_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)
    weights = None
    if args.balanced:
        counts = df[df[col] == "train"].category.value_counts().reindex(classes).to_numpy()
        weights = torch.tensor(1 / np.sqrt(counts), dtype=torch.float32, device=device)
        weights /= weights.mean()
    crit = nn.CrossEntropyLoss(weight=weights)

    best_f1, best_epoch, history = -1.0, -1, []
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y); n += len(y)
        vl, vy = run_inference(model, val_dl, device, amp)
        vm = summarize(vl, vy, classes, binary_map)
        history.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_acc": vm["accuracy"],
                        "val_macro_f1": vm["macro_f1"], "val_unsafe_recall": vm["binary"]["unsafe_recall"],
                        "sec": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch = vm["macro_f1"], epoch
            save_checkpoint(out / "best.pt", model, args.arch, classes, binary_map, size,
                            preprocessing=args.preprocessing)
        elif epoch - best_epoch >= tc["patience"]:
            print(f"early stop at epoch {epoch} (best {best_epoch})")
            break

    # Reload best, calibrate on val, evaluate on test.
    model.load_state_dict(torch.load(out / "best.pt", weights_only=False)["state_dict"])
    vl, vy = run_inference(model, val_dl, device, amp)
    cl, cy = run_inference(model, cal_dl, device, amp)
    temp = fit_temperature(cl, cy)
    save_checkpoint(out / "best.pt", model, args.arch, classes, binary_map, size, temperature=temp,
                    split=args.split, robust=args.robust, best_epoch=best_epoch,
                    preprocessing=args.preprocessing, balanced=args.balanced,
                    calibration_partition="cal" if args.manifest else "val")
    tl, ty = run_inference(model, test_dl, device, amp)
    np.savez(out / "logits.npz", val_logits=vl, val_targets=vy, test_logits=tl, test_targets=ty,
             cal_logits=cl, cal_targets=cy, classes=np.array(classes))
    rc = cfg["calibration"]
    metrics = {
        "arch": args.arch, "split": args.split, "robust": args.robust, "best_epoch": best_epoch,
        "val": summarize(vl, vy, classes, binary_map, temp, rc["reject_threshold"], rc["n_bins"]),
        "test": summarize(tl, ty, classes, binary_map, temp, rc["reject_threshold"], rc["n_bins"],
                          plot_prefix=out / "test", title=f"{run_name} test"),
        "history": history,
        "preprocessing": args.preprocessing, "balanced": args.balanced,
        "calibration_partition": "cal" if args.manifest else "val",
        "test_by_domain": {str(domain): summarize(tl[mask], ty[mask], classes, binary_map, temp,
                                  rc["reject_threshold"], rc["n_bins"])
                           for domain in df.loc[df[col] == "test", "domain"].unique()
                           for mask in [(df.loc[df[col] == "test", "domain"].to_numpy() == domain)]}
                          if "domain" in df else {},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    t = metrics["test"]
    print(f"[{run_name}] T={temp:.3f} test acc={t['accuracy']:.4f} macroF1={t['macro_f1']:.4f} "
          f"unsafe recall={t['binary']['unsafe_recall']:.4f} ECE raw={t['ece_raw']:.4f} cal={t['ece_calibrated']:.4f}")


if __name__ == "__main__":
    main()
