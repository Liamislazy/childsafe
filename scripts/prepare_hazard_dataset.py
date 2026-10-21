"""Validate and package independently annotated full frames for object training.

No downloads, pseudo masks, automatic labels, split generation or camera capture.
CSV: image,split,group_id,reviewed,annotation_complete,source,license,author,url.
Each image has one adjacent .txt in YOLO format; empty .txt = reviewed background.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]


def prepare(manifest, output, task='segment', classes=ROOT/'configs/hazard_classes.yaml'):
    manifest, output = Path(manifest).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('choose a new output directory; existing datasets are never overwritten')
    if task not in ('detect', 'segment'):
        raise ValueError('task must be detect or segment')
    names = yaml.safe_load(Path(classes).read_text(encoding='utf-8'))['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k):v for k,v in names.items()}
    if sorted(names) != list(range(len(names))) or len(set(names.values())) != len(names):
        raise ValueError('class IDs must be unique and contiguous from zero')
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError('no annotated images; the template is not training data')
    required={'image','split','group_id','reviewed','annotation_complete','source','license'}
    if not required.issubset(records[0]):
        raise ValueError('missing manifest columns: '+', '.join(sorted(required-set(records[0]))))
    groups, hashes, paths, prepared = {}, {}, set(), []
    counts = {split:{name:0 for name in names.values()} for split in ('train','val','test')}
    backgrounds = dict.fromkeys(counts, 0)
    for row in records:
        split, group = row['split'], row['group_id'].strip()
        if split not in counts or not group:
            raise ValueError('explicit train/val/test and nonempty group_id required')
        if any(str(row.get(k, '')).lower() not in ('true','1') for k in ('reviewed','annotation_complete')):
            raise ValueError('every frame needs reviewed, complete annotations, including background')
        if not row.get('source') or not row.get('license'):
            raise ValueError('source and license/consent basis required for every frame')
        if groups.setdefault(group, split) != split:
            raise ValueError('capture/person/source group crosses dataset splits')
        image = (manifest.parent/row['image']).resolve()
        if image in paths:
            raise ValueError('duplicate image path')
        paths.add(image)
        if not image.is_file() or not image.with_suffix('.txt').is_file():
            raise ValueError('each full-frame image needs its adjacent YOLO .txt label')
        with Image.open(image) as im:
            pixels = im.convert('RGB'); width,height = pixels.size
            digest = hashlib.sha256(f'{width}x{height}:'.encode()+pixels.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('duplicate decoded image content, including across splits')
        hashes[digest] = split
        text = '\n'.join(line.strip() for line in image.with_suffix('.txt').read_text(encoding='utf-8').splitlines() if line.strip())
        for line in text.splitlines():
            parts = line.split(); class_id = int(parts[0]); coords = list(map(float, parts[1:]))
            if class_id not in names or not all(0 <= v <= 1 for v in coords):
                raise ValueError('unknown class ID or invalid normalized coordinates')
            if task == 'detect':
                if len(coords) != 4:
                    raise ValueError('detection requires xc yc width height')
                x,y,w,h = coords
                if w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
                    raise ValueError('invalid or out-of-image bounding box')
            elif len(coords) < 6 or len(coords)%2:
                raise ValueError('segmentation needs polygons; boxes must never become fake masks')
            else:
                points = list(zip(coords[::2], coords[1::2]))
                area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2
                if area <= 1e-8:
                    raise ValueError('degenerate polygon')
            counts[split][names[class_id]] += 1
        if not text:
            backgrounds[split] += 1
        prepared.append((row,image,digest,text))
    for split in counts:
        if not any(row['split'] == split for row, *_ in prepared):
            raise ValueError('all three independent splits are required')
    # Do not create a 15-class artifact when only two classes have annotations.
    active = [name for name in names.values() if any(counts[s][name] for s in counts)]
    if not active:
        raise ValueError('no labeled target instances')
    missing = {s:[name for name in active if not counts[s][name]] for s in counts}
    if any(missing.values()):
        raise ValueError('active classes missing from an independent split: '+json.dumps(missing))
    compact = {names_id:active.index(name) for names_id,name in names.items() if name in active}
    output.mkdir(parents=True)
    exported = []
    for index,(row,image,digest,text) in enumerate(prepared):
        split = row['split']; stem = f'{index:06d}_{digest[:12]}'
        target = output/'images'/split/(stem+image.suffix.lower())
        label = output/'labels'/split/(stem+'.txt')
        target.parent.mkdir(parents=True,exist_ok=True); label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,target)
        lines = [str(compact[int(line.split()[0])])+' '+' '.join(line.split()[1:]) for line in text.splitlines()]
        label.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
        exported.append(dict(row,image=target.relative_to(output).as_posix(),sha256=digest))
    config = dict(path=str(output.resolve()), train='images/train',val='images/val',test='images/test',
                  names=dict(enumerate(active)))
    (output/'dataset.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    with (output/'manifest.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(exported[0])); writer.writeheader(); writer.writerows(exported)
    report = dict(task=task,classes=active,not_yet_annotated=[n for n in names.values() if n not in active],
                  images={s:sum(r['split']==s for r in exported) for s in counts},
                  instances={s:{n:counts[s][n] for n in active} for s in counts},
                  reviewed_background_images=backgrounds,
                  limitations=['Counts and exact duplicates checked; scene labels and near duplicates need human review.',
                               'Image test metrics are not camera event recall or clinical validation.'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--task', choices=['detect','segment'], default='segment')
    parser.add_argument('--classes', type=Path, default=ROOT/'configs/hazard_classes.yaml')
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest,args.out,args.task,args.classes),indent=2))
