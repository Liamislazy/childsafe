"""Check local assets and hashes without opening the camera or loading model weights."""
import argparse
import json
from pathlib import Path

from childsafe.config import FULL_DATASET_CHECKPOINT, ROOT, load_config
from childsafe.runtime.readiness import demo_readiness


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--ckpt', type=Path)
    p.add_argument('--out', type=Path, default=Path('outputs/piise_evidence/readiness.json'))
    args = p.parse_args(argv)
    defaults = [FULL_DATASET_CHECKPOINT,
                ROOT/'outputs/with_other/runs/efficientnet_b0_group_robust/best.pt',
                ROOT/'outputs/kaggle_v3/childsafe/outputs/runs/efficientnet_b0_group/best.pt',
                ROOT/'outputs/kaggle_v3/childsafe/outputs/runs/efficientnet_b0_group_robust/best.pt']
    ckpt = args.ckpt or next((path for path in defaults if path.is_file()), FULL_DATASET_CHECKPOINT)
    result = demo_readiness(load_config(args.config), ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(result['status'])
    print(args.out)
    return 2 if result['missing_models'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
