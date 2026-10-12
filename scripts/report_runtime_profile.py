"""Read numeric session features; no camera, audio or video capture."""
import argparse
import json
from pathlib import Path
import pandas as pd
from childsafe.evaluation.runtime_profile import runtime_profile

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('features',type=Path); p.add_argument('--max-gap',type=float,default=.25)
    p.add_argument('--out',type=Path,default=Path('outputs/runtime_profile.json'))
    a = p.parse_args(); result = runtime_profile(pd.read_csv(a.features),a.max_gap)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2,allow_nan=False))
