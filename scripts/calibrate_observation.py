"""Propose original-pixel thresholds from independently labeled adult/doll dev sessions."""
import argparse
import json
from pathlib import Path
import pandas as pd
from childsafe.config import load_config
from childsafe.evaluation.quality_calibration import propose_quality
from childsafe.evaluation.sessions import validate_sessions, config_hash, file_hash

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path); p.add_argument('--out',type=Path,default=Path('outputs/quality_proposal'))
    a = p.parse_args(); m = json.loads((a.session/'manifest.json').read_text(encoding='utf8'))
    validate_sessions([m]); cfg = load_config(a.session/'config.json')
    if config_hash(cfg) != m['config_sha256']: raise ValueError('session config changed')
    cfg,report = propose_quality(pd.read_csv(a.session/'features.csv'),
                                pd.read_csv(a.session/'observation_labels.csv'),cfg,m['split'])
    report.update(condition=m['condition'],recording_id=m['recording_id'],
                  input_features_sha256=file_hash(a.session/'features.csv'))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'proposed_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    (a.out/'dev_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
