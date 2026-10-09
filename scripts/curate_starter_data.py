"""Reproduce the initial contact-sheet review (2026-10-07).

This list is specific to the collected validation subset, not an automatic label reviewer.
Novel downloads remain pending. Ambiguous crops and multi-category source images are excluded.
"""
import json
import hashlib
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'data/supplement'
df=pd.read_csv(ROOT/'manifest.csv',dtype=str).fillna('')
snapshot=hashlib.sha256('\n'.join(sorted(df.category+':'+df.image_id)).encode()).hexdigest()
if snapshot != 'ac9a47b0b8bc8bf9bfe8678af2ca657c944f3b811c142044e66f96990954e144':
    raise ValueError('Collection changed: manual review required; this review list cannot approve new images')
ambiguous={'0a8fc50cd3e3df1a','2791837e97168d29','5ee08090de9f606a','e7bf630c163e9cbe',
           'f44cdd2942011550','89295272118b524e','bdb8abb449444489','0173241beb1633e9',
           '26f72a4eb15ea8c0','780a4baa7c50d6e8','e1ffa13fb120c12b'}
mixed=set(df.groupby('image_id').category.nunique().loc[lambda s:s>1].index)
df['reviewed']='false'
df['review_note']='excluded: ambiguous crop or multiple object categories in same source'
keep=~df.image_id.isin(ambiguous|mixed)
df.loc[keep,'reviewed']='true'
df.loc[keep,'review_note']='Contact-sheet visual review: recognizable object, no visible face; 2026-10-07'
df.to_csv(ROOT/'manifest.csv',index=False)
report=json.loads((ROOT/'collection_report.json').read_text())
report['reviewed']=int(keep.sum())
report['approved_by_class']=df.loc[keep].category.value_counts().to_dict()
report['excluded']=int((~keep).sum())
report['warning']='Small starter set reviewed via contact sheets, not a webcam validation set. Recheck labels before reuse.'
(ROOT/'collection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
