"""Inspect metadata only. Does not download or open audio/video."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from childsafe.data.audio_labels import audit_coughvid_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/choking-research/audio_label_audit.json"))
    args = parser.parse_args()
    data = args.metadata.read_bytes()
    report = audit_coughvid_metadata(pd.read_csv(args.metadata, dtype={"uuid": str}))
    report["input_sha256"] = hashlib.sha256(data).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(report["counts"], indent=2))
    print(f"Metadata audit written to {args.out}; clinical validation remains unestablished.")


if __name__ == "__main__":
    main()
