"""Build full-frame Pen/Chopsticks YOLO boxes from local Open Images metadata.

Unlike fetch_openimages.py, this preserves original frames and all target boxes.
Official train/validation/test splits stay separate. Images retain attribution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

import pandas as pd
from PIL import Image
import yaml

CLASSES = {'Pen': 'pen', 'Chopsticks': 'chopsticks'}


def yolo_box(row, class_id):
    x0,y0,x1,y1 = map(float,(row.XMin,row.YMin,row.XMax,row.YMax))
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('invalid normalized box')
    return f'{class_id} {(x0+x1)/2:.8f} {(y0+y1)/2:.8f} {x1-x0:.8f} {y1-y0:.8f}'


def build(meta, out, train_cap=150, eval_cap=50, workers=8):
    if (out/'dataset.yaml').exists():
        raise FileExistsError('completed dataset exists; use --out with a new directory')
    out.mkdir(parents=True,exist_ok=True)
    classes = pd.read_csv(meta/'class-descriptions-boxable.csv',header=None,names=['mid','name'])
    mids = {row.mid:list(CLASSES).index(row.name) for row in classes.itertuples() if row.name in CLASSES}
    if len(mids) != len(CLASSES): raise ValueError('required Open Images classes missing')
    seen = set(); records = []
    for source, split in [('train','train'),('validation','val'),('test','test')]:
        print('Reading annotations:',source,flush=True)
        filename = 'oidv6-train-annotations-bbox.csv' if source=='train' else f'{source}-annotations-bbox.csv'
        parts = []
        for chunk in pd.read_csv(meta/filename,chunksize=500000,
                usecols=['ImageID','LabelName','XMin','XMax','YMin','YMax','IsGroupOf','IsDepiction']):
            parts.append(chunk[chunk.LabelName.isin(mids)])
        boxes = pd.concat(parts,ignore_index=True)
        # Skip frames containing group annotations: they cannot be represented by
        # one box per object. Keep occluded/truncated instances with valid boxes.
        excluded = set(boxes.loc[(boxes.IsGroupOf!=0)|(boxes.IsDepiction!=0),'ImageID'])
        boxes = boxes[~boxes.ImageID.isin(excluded)]
        imagefile = 'train-images-boxable-with-rotation.csv' if source=='train' else f'{source}-images-with-rotation.csv'
        info = pd.read_csv(meta/imagefile,usecols=['ImageID','Rotation','License','Author','OriginalURL'])
        info = info[info.Rotation.fillna(0)==0].set_index('ImageID')
        boxes = boxes[boxes.ImageID.isin(info.index)]
        selected = set()
        for mid in mids:
            ids = sorted(set(boxes.loc[boxes.LabelName==mid,'ImageID']),
                         key=lambda x: hashlib.sha256(('childsafe-detector-42'+x).encode()).hexdigest())
            selected.update(ids[:train_cap if split=='train' else eval_cap])
        if seen & selected: raise ValueError('image ID crosses official splits')
        seen.update(selected)
        jobs = [(iid,g) for iid,g in boxes[boxes.ImageID.isin(selected)].groupby('ImageID')]
        print(split,'images',len(jobs),'boxes',sum(len(g) for _,g in jobs),flush=True)
        def fetch(job):
            iid, group = job
            path = out/'images'/split/f'{iid}.jpg'
            label = out/'labels'/split/f'{iid}.txt'
            path.parent.mkdir(parents=True,exist_ok=True); label.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                error = None
                for attempt in range(3):
                    try:
                        url=f'https://open-images-dataset.s3.amazonaws.com/{source}/{iid}.jpg'
                        with urllib.request.urlopen(url,timeout=30) as response:
                            payload=response.read()
                        with Image.open(io.BytesIO(payload)) as image:
                            image.verify()
                        temp=path.with_suffix('.part'); temp.write_bytes(payload); temp.replace(path)
                        break
                    except Exception as exc:
                        error=type(exc).__name__; time.sleep(attempt+1)
                if not path.exists(): return {'split':split,'image_id':iid,'status':error}
            with Image.open(path) as image: width,height=image.size
            label.write_text('\n'.join(yolo_box(r,mids[r.LabelName]) for r in group.itertuples())+'\n',encoding='utf-8')
            attribution=info.loc[iid]
            return dict(split=split,image_id=iid,status='ok',width=width,height=height,
                        boxes=len(group),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        license=attribution.License,author=attribution.Author,original_url=attribution.OriginalURL)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, record in enumerate(pool.map(fetch,jobs),1):
                records.append(record)
                if index%25==0 or index==len(jobs): print(split,index,'/',len(jobs),flush=True)
    pd.DataFrame(records).to_csv(out/'manifest.csv',index=False)
    successful = [r for r in records if r['status']=='ok']
    hashes = {}
    for record in successful:
        previous=hashes.setdefault(record['sha256'],record['split'])
        if previous!=record['split']: raise ValueError('identical image content crosses splits')
    if any(not any(r['split']==s for r in successful) for s in ['train','val','test']):
        raise ValueError('empty dataset split')
    config=dict(path=str(out.resolve()),train='images/train',val='images/val',test='images/test',
                names=dict(enumerate(CLASSES.values())))
    (out/'dataset.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    report=dict(images={s:sum(r['split']==s for r in successful) for s in ['train','val','test']},
                failed=len(records)-len(successful),classes=list(CLASSES.values()),
                annotation_type='bounding_boxes',split_policy='official Open Images splits')
    report['boxes_by_class'] = {}
    for split in ['train','val','test']:
        counts={name:0 for name in CLASSES.values()}
        for record in successful:
            if record['split'] != split: continue
            for line in (out/'labels'/split/f"{record['image_id']}.txt").read_text().splitlines():
                counts[list(CLASSES.values())[int(line.split()[0])]] += 1
        report['boxes_by_class'][split]=counts
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--meta',type=Path,default=Path('data/external/openimages/meta'))
    parser.add_argument('--out',type=Path,default=Path('data/detector/pen_chopsticks'))
    parser.add_argument('--train-cap',type=int,default=150)
    parser.add_argument('--eval-cap',type=int,default=50)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    build(args.meta,args.out,args.train_cap,args.eval_cap,args.workers)
