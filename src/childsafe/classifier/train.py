"""Train one Module A model on one split, then calibrate on val and evaluate on test.

    python -m childsafe.classifier.train --arch efficientnet_b0 --split group
    python -m childsafe.classifier.train --arch efficientnet_b0 --split random   # leakage comparison
    python -m childsafe.classifier.train --arch efficientnet_b0 --split group --robust  # A5

Outputs go to outputs/runs/<arch>_<split>[_robust]/.
"""
from __future__ import annotations

import argparse
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

    df = pd.read_csv(cfg["paths"]["splits"])
    col = f"split_{args.split}"
    classes = sorted(df["category"].unique())
    binary_map = dict(df[["category", "label"]].drop_duplicates().values)
    root = source_roots(cfg)   # ChildSUn + any extra sources
    size = tc["img_size"]

    def loader(part, tf, shuffle):
        ds = CatalogDataset(df[df[col] == part], root, classes, tf)
        sampler = None
        if shuffle and tc.get("balance") == "sqrt":
            # class-balanced sampling, weight ~ 1/sqrt(class count): rare classes are drawn more often
            # without fully flattening the distribution (one epoch still = len(ds) draws)
            counts = torch.bincount(ds.targets, minlength=len(classes)).float().clamp(min=1)
            sampler = torch.utils.data.WeightedRandomSampler(counts.rsqrt()[ds.targets], len(ds), replacement=True)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle and sampler is None, sampler=sampler,
                          num_workers=tc["num_workers"], pin_memory=device.type == "cuda",
                          persistent_workers=tc["num_workers"] > 0, drop_last=shuffle and len(ds) > bs)

    train_dl = loader("train", train_transform(size, args.robust), True)
    val_dl = loader("val", eval_transform(size), False)
    test_dl = loader("test", eval_transform(size), False)

    run_name = f"{args.arch}_{args.split}" + ("_robust" if args.robust else "")
    out = Path(args.out or Path(cfg["paths"]["outputs"]) / "runs" / run_name)
    out.mkdir(parents=True, exist_ok=True)

    lr = tc.get("lr_overrides", {}).get(args.arch, tc["lr"])
    print(f"[{run_name}] lr={lr} epochs={epochs} batch={bs}", flush=True)
    model = create_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)
    crit = nn.CrossEntropyLoss()

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
            save_checkpoint(out / "best.pt", model, args.arch, classes, binary_map, size)
        elif epoch - best_epoch >= tc["patience"]:
            print(f"early stop at epoch {epoch} (best {best_epoch})")
            break

    # Reload best, calibrate on val, evaluate on test.
    model.load_state_dict(torch.load(out / "best.pt", weights_only=False)["state_dict"])
    vl, vy = run_inference(model, val_dl, device, amp)
    temp = fit_temperature(vl, vy)
    save_checkpoint(out / "best.pt", model, args.arch, classes, binary_map, size, temperature=temp,
                    split=args.split, robust=args.robust, best_epoch=best_epoch)
    tl, ty = run_inference(model, test_dl, device, amp)
    np.savez(out / "logits.npz", val_logits=vl, val_targets=vy, test_logits=tl, test_targets=ty,
             classes=np.array(classes))
    rc = cfg["calibration"]
    metrics = {
        "arch": args.arch, "split": args.split, "robust": args.robust, "best_epoch": best_epoch,
        "val": summarize(vl, vy, classes, binary_map, temp, rc["reject_threshold"], rc["n_bins"]),
        "test": summarize(tl, ty, classes, binary_map, temp, rc["reject_threshold"], rc["n_bins"],
                          plot_prefix=out / "test", title=f"{run_name} test"),
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    t = metrics["test"]
    print(f"[{run_name}] T={temp:.3f} test acc={t['accuracy']:.4f} macroF1={t['macro_f1']:.4f} "
          f"unsafe recall={t['binary']['unsafe_recall']:.4f} ECE raw={t['ece_raw']:.4f} cal={t['ece_calibrated']:.4f}")


if __name__ == "__main__":
    main()
