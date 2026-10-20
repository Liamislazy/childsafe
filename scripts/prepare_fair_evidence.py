"""Export auditable comparison tables; opens no cameras and changes no thresholds."""
import argparse
from pathlib import Path

from childsafe.evaluation.fair_evidence import collect_evidence, write_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=Path, default=Path('outputs/scenarios'))
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--out', type=Path, default=Path('outputs/piise_evidence'))
    args = parser.parse_args(argv)
    bundle = collect_evidence(args.sessions, args.split)
    print(write_evidence(bundle, args.out))
    print('status:', bundle['status'])


if __name__ == '__main__':
    main()
