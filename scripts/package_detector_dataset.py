"""Package full-frame detector dataset, attribution and backbone for Kaggle."""
import argparse
from pathlib import Path
import zipfile


def package(source,target):
    required=['dataset.yaml','manifest.csv','report.json','images/train','images/val','images/test']
    if any(not (source/name).exists() for name in required): raise ValueError('dataset preparation incomplete')
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix in ('.jpg','.txt','.yaml','.csv','.json'):
                archive.write(path,'pen_chopsticks/'+path.relative_to(source).as_posix())
        archive.write('models/yolo11n-seg.pt','models/yolo11n-seg.pt')
    print(target.resolve(),target.stat().st_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('data/detector/pen_chopsticks'))
    parser.add_argument('--out',type=Path,default=Path('outputs/childsafe_pen_chopsticks_detection.zip'))
    args=parser.parse_args(); package(args.source,args.out)
