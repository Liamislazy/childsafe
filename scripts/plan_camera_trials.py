"""Create a targeted adult/doll trial checklist, not labeled observations or results."""
import argparse
import csv
import random
from pathlib import Path

# Controlled cases cover known failures without an unmanageable full factorial.
CASES=[
    ('horizontal','720p','bright','front',1,'none','oversized_long_proxy','approach'),
    ('elevated','720p','bright','front',1,'none','oversized_long_proxy','approach'),
    ('elevated','480p','bright','front',1,'none','oversized_small_part_proxy','approach'),
    ('elevated','1080p','bright','front',1,'none','oversized_small_part_proxy','approach'),
    ('elevated','720p','dim','front',1,'none','oversized_long_proxy','approach'),
    ('elevated','720p','bright','profile',1,'none','oversized_long_proxy','approach'),
    ('elevated','720p','bright','back',1,'object','oversized_long_proxy','approach'),
    ('elevated','720p','bright','front',1,'hand','oversized_long_proxy','brief_occlusion'),
    ('elevated','720p','bright','front',1,'object','oversized_long_proxy','long_occlusion'),
    ('elevated','720p','bright','front',2,'none','oversized_long_proxy','independent_actions'),
    ('elevated','720p','bright','front',2,'person','oversized_long_proxy','crossing'),
    ('elevated','720p','bright','front',2,'none','oversized_long_proxy','person_replacement'),
    ('elevated','720p','bright','front',4,'none','oversized_long_proxy','stress'),
    ('horizontal','720p','bright','front',1,'none','none','scratch_face'),
    ('elevated','720p','bright','front',1,'none','none','rest_chin'),
    ('elevated','720p','bright','front',1,'none','cup_proxy','ordinary_action'),
    ('elevated','720p','bright','front',1,'none','oversized_toy_proxy','approach'),
    ('elevated','720p','bright','front',1,'none','unknown_proxy','approach'),
    ('elevated','720p','bright','front',1,'none','oversized_long_proxy','camera_disconnect'),
    ('elevated','720p','bright','front',1,'none','oversized_long_proxy','object_in_front_of_face_without_contact'),
]


def plan(repeats=1,seed=42):
    if repeats<1: raise ValueError('repeats must be positive')
    rows=[]
    for split in ('dev','test'):
        for repeat in range(repeats):
            for i,case in enumerate(CASES,1):
                row=dict(zip(('angle','resolution','lighting','pose','people','occlusion','prop','action'),case))
                row.update(trial_id=f'{split}-{repeat+1:02d}-{i:02d}',split=split,condition='-'.join(map(str,case)),
                           subject_ids='',consent_confirmed='',status='planned_not_measured',
                           negative_seconds_observed='',positive_events_observed='',notes='')
                rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats',type=int,default=1); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path,default=Path('outputs/planned_trials.csv'))
    a=p.parse_args(argv); rows=plan(a.repeats,a.seed)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(a.out)


if __name__=='__main__': main()
