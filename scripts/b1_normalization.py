"""B1: which scale normalizer keeps the normalized distance most stable?

Protocol: hold the hand (or mock object) at a FIXED physical distance from the mouth
(e.g. touching the chin, or 10 cm away with a ruler), and record short sessions while
changing only camera distance and head yaw:

    python scripts/run_webcam.py --no-classifier --log-features --condition fixed10cm_cam60_yaw0
    python scripts/run_webcam.py --no-classifier --log-features --condition fixed10cm_cam100_yaw30
    ...
Then:
    python scripts/b1_normalization.py --glob "outputs/features/live_fixed10cm_*.csv"

For each normalizer we report the coefficient of variation (std/mean) of the per-condition
median distance across conditions (lower = more stable), plus the within-condition CV (jitter).
"""
import argparse
import glob

import numpy as np
import pandas as pd

from childsafe.vision.geometry import NORMALIZERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(args.glob)], ignore_index=True)
    df = df[(df.face_ok == 1) & (df.hand_ok == 1)]
    print(f"{len(df)} frames, conditions: {sorted(df.condition.unique())}\n")
    print("| normalizer | median per condition | CV across conditions | mean CV within condition |")
    print("|---|---|---|---|")
    for n in NORMALIZERS:
        col = f"d_{n}"
        med = df.groupby("condition")[col].median()
        within = df.groupby("condition")[col].agg(lambda x: x.std() / x.mean() if x.mean() else np.nan)
        print(f"| {n} | {', '.join(f'{k}:{v:.2f}' for k, v in med.items())} | {med.std() / med.mean():.3f} | {within.mean():.3f} |")
    print("\nAlso check raw pixels (d_px) as a sanity reference:",
          f"CV across conditions = {df.groupby('condition').d_px.median().pipe(lambda m: m.std() / m.mean()):.3f}")


if __name__ == "__main__":
    main()
