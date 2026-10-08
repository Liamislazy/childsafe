"""Merge reviewed supplemental crops, preserve canonical test, audit duplicates before training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from childsafe.data.dedup import compute_hashes, find_duplicate_pairs

def merge_manifest(base, raw_root, supplement, seed=42, audit=True):
    base = base.copy()
    base['path'] = [str((Path(raw_root)/p).resolve()) for p in base.path]
    base['group'] = 'childsun:' + base.group.astype(str)
    base['domain'] = 'childsun'
    base['reviewed'] = True
    # Reserve half the existing validation groups for temperature fitting.
    rng = np.random.default_rng(seed)
    for category, part in base[base.split_group == 'val'].groupby('category'):
        groups = sorted(part.group.unique()); rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError(f'Need two validation groups for {category}')
        base.loc[base.group.isin(groups[:max(1,len(groups)//2)]), 'split_group'] = 'cal'
    source = Path(supplement)
    extra = pd.read_csv(source).fillna('')
    approved = extra.reviewed.astype(str).str.lower().isin(['true','1','yes'])
    print(f'Review: {approved.sum()} approved, {(~approved).sum()} pending/excluded', flush=True)
    extra = extra[approved].copy()
    if extra.empty:
        raise ValueError('Review manifest.csv first: set reviewed=true only for verified object labels and privacy')
    if not set(extra.label).issubset({'safe','unsafe'}):
        raise ValueError('Unknown samples need a separately designed rejection evaluation; never map them to safe')
    extra['path'] = [str((source.parent/p).resolve()) for p in extra.path]
    extra['split_group'] = ''
    df = pd.concat([base, extra], ignore_index=True)
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i,j):
        parent[find(j)] = find(i)
    for _, rows in df.groupby('group'):
        ids=list(rows.index)
        for i in ids[1:]: union(ids[0],i)
    # Decode every image; fail instead of silently hashing corrupt files as duplicates.
    pixels = {}
    for i, path in enumerate(df.path):
        with Image.open(path) as image:
            image = image.convert('RGB')
            digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        if digest in pixels:
            j=pixels[digest]
            if df.at[i,'category'] != df.at[j,'category']:
                raise ValueError('Identical image has conflicting category labels; review annotations')
            union(i,j)
        else: pixels[digest]=i
    duplicate_count = 0
    if audit:
        ph,dh=compute_hashes(df.path.tolist())
        pairs=find_duplicate_pairs(ph,dh,8,5)
        duplicate_count=len(pairs)
        for i,j,*_ in pairs:
            i,j=int(i),int(j)
            if df.at[i,'category'] != df.at[j,'category']:
                raise ValueError(f'Near-duplicate crops have conflicting categories: {df.at[i,"path"]}, {df.at[j,"path"]}')
            union(i,j)
    df['group'] = [f'merged:{find(i)}' for i in range(len(df))]
    for _, rows in df.groupby('group'):
        roles = set(rows.split_group)-{''}
        if len(roles)>1:
            raise ValueError(f'Duplicate links canonical partitions: {rows.path.tolist()[:4]}; review before training')
        if roles: df.loc[rows.index,'split_group']=next(iter(roles))
    for category in sorted(df.loc[df.split_group=='','category'].unique()):
        part=df[(df.split_group=='') & (df.category==category)]
        groups=sorted(part.group.unique()); rng.shuffle(groups)
        if category not in set(base.category) and len(groups)<10:
            raise ValueError(f'{category}: need >=10 independent source groups; got {len(groups)}')
        # With sparse supplements of existing classes, keep all in train (base holdouts remain).
        n=len(groups)
        n_hold=max(1,int(n*.1)) if n>=10 else 0
        for role, subset in [('test',groups[:n_hold]),('cal',groups[n_hold:2*n_hold]),
                             ('val',groups[2*n_hold:3*n_hold]),('train',groups[3*n_hold:])]:
            df.loc[df.group.isin(subset), 'split_group']=role
    if df.groupby('group').split_group.nunique().max()!=1:
        raise ValueError('Group leakage')
    return df, {'images':len(df), 'near_duplicate_pairs':duplicate_count, 'seed':seed,
                'counts':df.groupby(['category','domain','split_group']).size().to_dict()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base', default='data/splits/splits.csv')
    ap.add_argument('--raw-root', required=True)
    ap.add_argument('--supplement', required=True)
    ap.add_argument('--out', default='data/splits/expanded.csv')
    args=ap.parse_args()
    df,report=merge_manifest(pd.read_csv(args.base),args.raw_root,args.supplement)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    report['counts']={' / '.join(k):int(v) for k,v in report['counts'].items()}
    out.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(df.groupby(['category','split_group']).size().unstack(fill_value=0).to_string())

if __name__=='__main__': main()
