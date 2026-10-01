"""E1 / E4 / D3 tuning on logged features + annotations.

Inputs
  outputs/features/<video>.csv          from extract_features.py
  data/scenarios/annotations.csv        video,start,end,label   (seconds; ELAN/CVAT export)
  data/scenarios/videos.csv             video,split             (split = dev | test)

    python scripts/evaluate_alerts.py report   --split test     # full system vs distance-only baseline + ablations
    python scripts/evaluate_alerts.py tradeoff --split test     # FA/h vs recall curves
    python scripts/evaluate_alerts.py grid     --split dev      # tune D3 thresholds (never on test)
"""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from childsafe.alerts.processor import CAUTION, DANGER, Components
from childsafe.config import DEFAULT_CONFIG, ROOT, load_config
from childsafe.evaluation.events import distance_only_levels, event_metrics, pool
from childsafe.evaluation.replay import replay

ABLATIONS = ["smoothing", "hysteresis", "n_of_m", "cascade", "object_gate", "calibration",
             "lost_landmark", "approach", "cooldown"]


def load_data(args, cfg):
    feats_dir = Path(args.features or Path(cfg["paths"]["outputs"]) / "features")
    ann = pd.read_csv(args.annotations)
    vids = pd.read_csv(args.videos) if Path(args.videos).exists() else None
    names = sorted(ann["video"].unique())
    if vids is not None and args.split != "all":
        names = [v for v in names if v in set(vids[vids["split"] == args.split]["video"])]
    data = []
    for v in names:
        f = feats_dir / f"{v}.csv"
        if not f.exists():
            print(f"[skip] no features for {v}")
            continue
        data.append((v, pd.read_csv(f), ann[ann["video"] == v]))
    if not data:
        raise SystemExit("no videos with both features and annotations")
    return data


def evaluate_system(data, cfg, components=None, min_level=DANGER, tol=2.0):
    res = []
    for v, feats, gt in data:
        out = replay(feats, cfg, components)
        dur = feats["t"].iloc[-1] - feats["t"].iloc[0] + 1 / feats["fps"].iloc[0] if "fps" in feats else None
        res.append(event_metrics(out["t"].to_numpy(), out["level"].to_numpy(), gt, dur, min_level, tol,
                                 cfg["temporal"]["merge_s"]))
    return pool(res)


def evaluate_baseline(data, cfg, threshold, tol=2.0):
    """Per-frame distance-only baseline: no smoothing, no temporal rule, no object, no merging."""
    norm = cfg["landmarks"]["normalizer"]
    res = []
    for v, feats, gt in data:
        lv = distance_only_levels(feats[f"d_{norm}"].to_numpy(), threshold)
        dur = feats["t"].iloc[-1] - feats["t"].iloc[0] + 1 / feats["fps"].iloc[0] if "fps" in feats else None
        res.append(event_metrics(feats["t"].to_numpy(), lv, gt, dur, DANGER, tol, merge_s=0.0))
    return pool(res)


def _row(name, m):
    return (f"| {name} | {m['event_recall']:.3f} | {m['false_alarms_per_hour']:.1f} | {m['latency_median_s']:.2f} | "
            f"{m['latency_p90_s']:.2f} | {m['event_count_error']:+d} | {m['per_second_agreement']:.3f} |")


HEADER = ["| system | event recall | false alarms / h | latency median (s) | latency p90 (s) | count error | per-second agreement |",
          "|---|---|---|---|---|---|---|"]


def cmd_report(args, cfg, data):
    lines = [f"# Alert evaluation ({args.split} split, tolerance {args.tol}s)\n",
             f"Videos: {len(data)}. Hours: {sum(len(f) / f['fps'].iloc[0] for _, f, _ in data) / 3600:.2f}\n"]
    for lvl, name in ((DANGER, "danger"), (CAUTION, "caution or danger")):
        lines += [f"\n## Alerts at level: {name}\n"] + HEADER
        lines.append(_row("full system", evaluate_system(data, cfg, None, lvl, args.tol)))
        if lvl == DANGER:
            lines.append(_row(f"distance-only baseline (d < {cfg['temporal']['d_enter']})",
                              evaluate_baseline(data, cfg, cfg["temporal"]["d_enter"], args.tol)))
        for comp in ABLATIONS:
            lines.append(_row(f"without {comp}", evaluate_system(data, cfg, Components.from_off([comp]), lvl, args.tol)))
    text = "\n".join(lines)
    out = Path(cfg["paths"]["docs"]) / f"results_alerts_{args.split}.md"
    out.write_text(text, encoding="utf-8")
    print(text)


def cmd_tradeoff(args, cfg, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ths = np.round(np.arange(0.15, 0.85, 0.05), 2)
    sys_pts, base_pts = [], []
    for th in ths:
        c = copy.deepcopy(cfg)
        c["temporal"]["d_enter"] = float(th)
        c["temporal"]["d_exit"] = float(th + (cfg["temporal"]["d_exit"] - cfg["temporal"]["d_enter"]))
        m = evaluate_system(data, c, None, DANGER, args.tol)
        b = evaluate_baseline(data, cfg, float(th), args.tol)
        sys_pts.append((th, m["false_alarms_per_hour"], m["event_recall"]))
        base_pts.append((th, b["false_alarms_per_hour"], b["event_recall"]))
        print(f"th={th}: system FA/h={m['false_alarms_per_hour']:.1f} R={m['event_recall']:.3f} | "
              f"baseline FA/h={b['false_alarms_per_hour']:.1f} R={b['event_recall']:.3f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    for pts, lab in ((sys_pts, "full system"), (base_pts, "distance-only per frame")):
        ax.plot([p[1] for p in pts], [p[2] for p in pts], marker="o", label=lab)
    ax.set_xlabel("false alarms per hour (non-dangerous time)")
    ax.set_ylabel("event recall")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path(cfg["paths"]["outputs"]) / f"tradeoff_{args.split}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    pd.DataFrame(sys_pts, columns=["threshold", "fa_per_h", "recall"]).assign(system="full").pipe(
        lambda d: pd.concat([d, pd.DataFrame(base_pts, columns=["threshold", "fa_per_h", "recall"]).assign(system="baseline")])
    ).to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out}")


def cmd_grid(args, cfg, data):
    if args.split == "test":
        raise SystemExit("do not tune on the test split")
    grid = {"d_enter": [0.25, 0.3, 0.35, 0.4, 0.5], "n": [2, 3, 5], "p_unsafe_danger": [0.5, 0.6, 0.7, 0.8],
            "dwell_danger_s": [0.0, 0.3, 0.5, 1.0]}
    gap = cfg["temporal"]["d_exit"] - cfg["temporal"]["d_enter"]
    results = []
    for d_enter, n, p, dwell in itertools.product(*grid.values()):
        c = copy.deepcopy(cfg)
        c["temporal"].update(d_enter=d_enter, d_exit=d_enter + gap, n_of_m=[n, cfg["temporal"]["n_of_m"][1]])
        c["rules"].update(p_unsafe_danger=p, dwell_danger_s=dwell)
        m = evaluate_system(data, c, None, DANGER, args.tol)
        results.append({"d_enter": d_enter, "n": n, "p_unsafe_danger": p, "dwell_danger_s": dwell,
                         "recall": m["event_recall"], "fa_per_h": m["false_alarms_per_hour"],
                         "latency_median_s": m["latency_median_s"]})
    df = pd.DataFrame(results)
    df.to_csv(Path(cfg["paths"]["outputs"]) / "grid_dev.csv", index=False)
    ok = df[df["recall"] >= args.min_recall]
    if ok.empty:
        print(f"no setting reaches recall >= {args.min_recall}; best recall {df.recall.max():.3f}")
        best = df.sort_values(["recall", "fa_per_h"], ascending=[False, True]).iloc[0]
    else:
        best = ok.sort_values(["fa_per_h", "latency_median_s"]).iloc[0]
    print("best:", best.to_dict())
    tuned = {"_base_": Path(args.config).resolve().as_posix() if args.config else DEFAULT_CONFIG.as_posix(),
             "temporal": {"d_enter": float(best.d_enter), "d_exit": float(best.d_enter + gap),
                          "n_of_m": [int(best.n), cfg["temporal"]["n_of_m"][1]]},
             "rules": {"p_unsafe_danger": float(best.p_unsafe_danger), "dwell_danger_s": float(best.dwell_danger_s)}}
    out = Path(cfg["paths"]["outputs"]) / "tuned_config.yaml"
    out.write_text(yaml.safe_dump(tuned), encoding="utf-8")
    print(f"wrote {out}; then run once: evaluate_alerts.py report --split test --config {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "tradeoff", "grid"])
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--annotations", default=str(ROOT / "data" / "scenarios" / "annotations.csv"))
    ap.add_argument("--videos", default=str(ROOT / "data" / "scenarios" / "videos.csv"))
    ap.add_argument("--tol", type=float, default=2.0, help="event matching tolerance, seconds (handoff: 1-3 s)")
    ap.add_argument("--min-recall", type=float, default=0.9, help="grid: minimum dev recall, then minimise FA/h")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = load_data(args, cfg)
    {"report": cmd_report, "tradeoff": cmd_tradeoff, "grid": cmd_grid}[args.cmd](args, cfg, data)


if __name__ == "__main__":
    main()
