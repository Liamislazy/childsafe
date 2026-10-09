"""Where does the detection chain break? Reads a camera_session.csv (downloaded from the web demo,
numbers only, no images) and prints a funnel from 'person found' to 'unsafe object confirmed at mouth'.

    python scripts/diagnose_session.py %USERPROFILE%\\Downloads\\camera_session.csv

Suggested 1-minute recording (adult volunteer, blunt/plastic props, near the mouth only):
10 s nothing in hand, then ~10 s each: hand at mouth, spoon, fork, a pen, phone.
"""
import argparse
import ast
from collections import Counter

import pandas as pd

from childsafe.config import load_config


def truthy(s):
    return s.astype(str).str.lower().isin(['true', '1', '1.0'])


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else f"{a}/0"


def reasons(df):
    c = Counter()
    for v in df.get('quality_reasons', pd.Series(dtype=str)).dropna():
        try:
            c.update(ast.literal_eval(v) if isinstance(v, str) else v)
        except (ValueError, SyntaxError):
            c[str(v)] += 1
    return c.most_common(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    args = ap.parse_args()
    cfg = load_config()
    df = pd.read_csv(args.csv)
    df = df[df.get('person_id', 0).fillna(0) > 0] if 'person_id' in df else df
    n = len(df)
    if not n:
        raise SystemExit('No person rows: YOLO never found a person. Check lighting/framing first.')
    duration = df['t'].max() - df['t'].min() if 't' in df else float('nan')
    print(f"{n} person-frames over {duration:.1f} s (~{n / max(duration, 1e-9):.1f} fps)\n")

    face = truthy(df['face_ok']) if 'face_ok' in df else pd.Series(False, index=df.index)
    mouth = truthy(df['mouth_visible'])
    hand = truthy(df['hand_ok'])
    print("1) Landmarks")
    print("   face found        ", pct(int(face.sum()), n))
    print("   mouth usable      ", pct(int(mouth.sum()), n), "<- after blur/light/head-angle/size checks")
    print("   hand found        ", pct(int(hand.sum()), n))
    lost = df[face & ~mouth]
    if len(lost):
        print("   why a found face was not usable:", reasons(lost))

    d = pd.to_numeric(df.get('d_face_width'), errors='coerce')
    near_hand = d <= cfg['temporal']['d_exit']
    near = df[near_hand]
    print(f"\n2) Frames with a hand near the mouth (d <= {cfg['temporal']['d_exit']} face widths): {len(near)}")
    if len(near):
        seen = pd.to_numeric(near.get('scene_objects'), errors='coerce').fillna(0) > 0
        assigned = truthy(near['object_present'])
        mask = truthy(near['mask_reliable']) if 'mask_reliable' in near else assigned & False
        prox = truthy(near['proximity_evidence'])
        cand = pd.to_numeric(near.get('candidate_object_d_relative'), errors='coerce') <= cfg['temporal']['d_enter']
        print("   YOLO saw any object       ", pct(int(seen.sum()), len(near)))
        print("   object linked to the hand ", pct(int(assigned.sum()), len(near)))
        print("   object mask confident     ", pct(int(mask.sum()), len(near)),
              f"(>= {cfg['quality']['min_mask_conf']}, needed for DANGER)")
        print("   confirmed mask at mouth   ", pct(int(prox.sum()), len(near)))
        print("   candidate mask near mouth ", pct(int(cand.sum()), len(near)), "(CAUTION only)")
        labels = Counter(l.split(':')[0] for v in near.get('scene_object_labels', pd.Series(dtype=str)).dropna()
                         for l in str(v).split(';') if l)
        print("   YOLO labels near mouth    ", labels.most_common(8) or 'none')
        cls = near[pd.to_numeric(near.get('p_unsafe_cal'), errors='coerce').notna()]
        if len(cls):
            print("   classifier said           ", Counter(cls['obj_class']).most_common(6))
        else:
            print("   classifier said            never ran (it only runs on a YOLO object linked to the hand)")
        print("   other reasons             ", reasons(near))

    print("\nReading it: the first line with a low percentage is where alerts are lost.")


if __name__ == '__main__':
    main()
