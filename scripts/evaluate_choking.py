"""Evaluate anonymous metadata from a future choking model; never read videos/audio."""
import argparse
import json
from pathlib import Path
import pandas as pd
from childsafe.evaluation.choking import choking_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--recordings', required=True)
    ap.add_argument('--out', default='outputs/choking_evaluation.json')
    args = ap.parse_args()
    frames = pd.read_csv(args.predictions, dtype={'recording_id': str, 'subject_id': str})
    annotations = pd.read_csv(args.annotations, dtype={'recording_id': str, 'subject_id': str}, keep_default_na=False)
    recordings = pd.read_csv(args.recordings, dtype={'recording_id': str})
    report = choking_report(frames, annotations, recordings)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Wrote {path}. This evaluator does not establish clinical validity.')


if __name__ == '__main__': main()
