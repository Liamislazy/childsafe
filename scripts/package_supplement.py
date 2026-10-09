"""Package crops + attribution for Kaggle; omit large downloaded annotation tables."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/supplement'
dest=ROOT/'outputs/datasets/childsafe_supplement.zip'
dest.parent.mkdir(parents=True,exist_ok=True)
with ZipFile(dest,'w',ZIP_DEFLATED) as archive:
    for folder in ['images','review']:
        for p in sorted((source/folder).rglob('*')):
            if p.is_file(): archive.write(p,p.relative_to(source).as_posix())
    for name in ['manifest.csv','collection_report.json']:
        archive.write(source/name,name)
    archive.write(ROOT/'docs/kaggle_expanded_training.md','README.md')
print(f'{dest} ({dest.stat().st_size/1e6:.1f} MB)')
