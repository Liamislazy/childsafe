"""One scoring definition for the product report, threshold search and trade-off curve."""
import argparse
import copy
import json
from pathlib import Path

from childsafe.config import load_config
from childsafe.evaluation.product_report import METRIC_VERSION,pool_reports
from childsafe.evaluation.session_reports import load_records,build_report,score,tune


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd',choices=['report','grid','tradeoff'])
    p.add_argument('--sessions',type=Path,required=True)
    p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--min-recall',type=float,default=.9)
    a=p.parse_args(argv)
    if a.cmd in ('grid','tradeoff') and a.split!='dev':
        p.error('threshold exploration is dev-only; evaluate the frozen setting once on test')
    records=load_records(a.sessions,a.split)
    override=load_config(a.config) if a.config else None
    out=a.out or Path('outputs')/f'{a.cmd}_{a.split}.json'
    if a.cmd=='report':
        result=build_report(records,override)
    elif a.cmd=='grid':
        if override is not None: p.error('grid starts from the recorded configuration')
        cfg,result=tune(records,a.split,a.min_recall)
        write_json(out.with_name(out.stem+'_config.json'),cfg)
    else:
        points=[]
        for threshold in (.2,.25,.3,.35,.4,.45,.5):
            parts=[]; baselines=[]
            for r in records:
                cfg=copy.deepcopy(override or r.cfg)
                gap=cfg['temporal']['d_exit']-cfg['temporal']['d_enter']
                cfg['temporal'].update(d_enter=threshold,d_exit=threshold+gap)
                parts.append(score(r,cfg)); baselines.append(score(r,cfg,baseline=True))
            points.append(dict(threshold=threshold,system=pool_reports(parts),baseline=pool_reports(baselines)))
        result=dict(split='dev',metric_definition=METRIC_VERSION,points=points)
    write_json(out,result)
    print(out)
