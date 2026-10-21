"""Package reviewed detector data and explicit local backbone; do not upload anything."""
import argparse
import json
from pathlib import Path
import zipfile


def package(source, target, backbone=Path('models/yolo11n-seg.pt')):
    source,target,backbone = Path(source),Path(target),Path(backbone)
    if target.exists():
        raise FileExistsError('choose a new ZIP path')
    for name in ('dataset.yaml','manifest.csv','report.json','images/train','images/val','images/test'):
        if not (source/name).exists():
            raise ValueError('reviewed dataset is incomplete: '+name)
    if not backbone.is_file():
        raise FileNotFoundError('local backbone missing')
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if not report.get('classes') or report.get('task') not in ('detect','segment'):
        raise ValueError('invalid dataset report')
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_STORED) as archive:
        for folder in ('images','labels'):
            for path in sorted((source/folder).rglob('*')):
                if path.is_file():
                    archive.write(path,'hazard_dataset/'+path.relative_to(source).as_posix())
        for name in ('dataset.yaml','manifest.csv','report.json'):
            archive.write(source/name,'hazard_dataset/'+name)
        archive.write(backbone,'models/yolo11n-seg.pt')
    return target.stat().st_size


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--backbone',type=Path,default=Path('models/yolo11n-seg.pt'))
    args=parser.parse_args()
    print('ZIP bytes:',package(args.source,args.out,args.backbone))
