"""Compatibility entrypoint for the shared, strict session reporting pipeline."""
import argparse
from pathlib import Path
from childsafe.evaluation.cli import main


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--out',type=Path,default=Path('outputs/product_report.json'))
    p.add_argument('--config',type=Path)
    a=p.parse_args()
    args=['report','--sessions',str(a.root),'--split',a.split,'--out',str(a.out)]
    if a.config: args+=['--config',str(a.config)]
    main(args)
