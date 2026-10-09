"""Build standalone notebooks for the complete downloaded training dataset."""
import json
import base64
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': s.splitlines(keepends=True)}


def build_notebook(filename='childsafe_expanded_kaggle.ipynb'):
    files = ['configs/default.yaml', 'configs/with_other.yaml', 'configs/with_other_train.yaml',
             'scripts/prepare_full_training.py', 'data/splits/splits_with_other_train.csv']
    files += [p.relative_to(ROOT).as_posix() for folder in ('src/childsafe/data', 'src/childsafe/classifier')
              for p in sorted((ROOT/folder).glob('*.py'))]
    files += ['src/childsafe/__init__.py', 'src/childsafe/config.py']
    cells = [md('''# ChildSafe: full dataset training on Kaggle
Train an 11-class classifier using the complete pinned pool: 5,353 ChildSUn images,
20,557 Open Images train crops and 1,973 validation crops (27,883 total).
The pool is partitioned into train/validation/calibration/test; not every image enters gradient updates.
Open Images test (4,069 crops) stays outside this pool for external evaluation.

1. Add ChildSUn and the Open Images crops as Kaggle inputs. Keep train/, validation/, test/
   and manifest.csv together. Upload crops and attribution manifest; metadata CSVs are unnecessary.
2. Enable GPU and Internet; edit RAW_ROOT and OPENIMAGES_ROOT below.
3. Run setup and preparation. The notebook validates every source against the pinned split,
   preserves all original test rows, and reserves validation groups for calibration.
4. Train the selected variant, then evaluate it. Download full_dataset_results.zip.

`other` is a rejection class, not a safety claim. Pen is currently in `other`; coin is not
a separate trained class. The older reviewed pen/coin supplement is a separate experiment.
Classifier scores do not validate elevated-camera alerts or locate pens/coins.
'''), code('''%pip install -q timm imagehash scikit-learn pandas scipy matplotlib pyyaml tqdm pillow
import os, sys, subprocess, json, shutil
from pathlib import Path
import pandas as pd
import yaml
from IPython.display import display, FileLink

REPO = Path('/kaggle/working/childsafe')
REPO.mkdir(exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO/'src'))
os.environ['PYTHONPATH'] = str(REPO/'src')
for folder in ['configs', 'scripts', 'src/childsafe/data', 'src/childsafe/classifier',
               'data/splits', 'outputs/full_dataset']:
    (REPO/folder).mkdir(parents=True, exist_ok=True)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'Enable GPU before the full training run'
'''), md('## Embedded project source and pinned split')]
    for name in files:
        if name.endswith('.csv'):
            payload = base64.b64encode(gzip.compress((ROOT/name).read_bytes(), mtime=0)).decode('ascii')
            cells.append(code("import base64, gzip\n" +
                              f"Path({name!r}).write_bytes(gzip.decompress(base64.b64decode({payload!r})))\n"))
        else:
            cells.append(code(f'%%writefile {name}\n' + ((ROOT/name).read_text(encoding='utf-8') or '# package\n')))
    cells += [md('''## Input paths and full-data validation
Use the input panel to find these exact roots. RAW_ROOT directly contains canonical
ChildSUn filenames. OPENIMAGES_ROOT directly contains train/, validation/, test/, manifest.csv.
The program stops on missing or extra training files rather than silently training a subset.
The existing near-duplicate/source groups are reused; test identities stay fixed.
'''), code('''RAW_ROOT = Path('/kaggle/input/CHANGE-ME-CHILDSUN/CHANGE-ME-IMAGE-FOLDER')
OPENIMAGES_ROOT = Path('/kaggle/input/CHANGE-ME-OPENIMAGES/openimages')
assert RAW_ROOT.is_dir(), RAW_ROOT
assert (OPENIMAGES_ROOT/'manifest.csv').is_file(), OPENIMAGES_ROOT
download = pd.read_csv(OPENIMAGES_ROOT/'manifest.csv')
assert len(download[(download['split'] == 'test') & download.status.isin(['ok', 'exists'])]) == 4069, 'Upload all external test crops'

MANIFEST = Path('data/splits/full_training.csv')
CONFIG = Path('configs/kaggle_full.yaml')
subprocess.run([sys.executable, 'scripts/prepare_full_training.py',
                '--raw-root', str(RAW_ROOT), '--openimages-root', str(OPENIMAGES_ROOT),
                '--out', str(MANIFEST)], check=True)
cfg = {'_base_': 'with_other_train.yaml',
       'paths': {'raw_data': str(RAW_ROOT), 'splits': str(MANIFEST),
                 'outputs': 'outputs/full_dataset'},
       'extra_sources': {'openimages': str(OPENIMAGES_ROOT/'validation'),
                         'openimages_train': str(OPENIMAGES_ROOT/'train')},
       'train': {'num_workers': 2, 'batch_size': 32, 'balance': 'sqrt'}}
CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
df = pd.read_csv(MANIFEST)
assert len(df) == 27883 and df.category.nunique() == 11
assert df.groupby('group').split_group.nunique().max() == 1
display(df.groupby(['source', 'split_group']).size().unstack(fill_value=0))
display(df.groupby(['category', 'split_group']).size().unstack(fill_value=0))
'''), md('''## Train one selected model
Default: EfficientNet-B0, robust augmentation, letterbox, sqrt balanced sampling,
up to 30 epochs and early stopping on validation macro-F1. Calibration uses only cal.
Use validation results to choose a variant before interpreting held-out test.
Sampling alone balances training; do not also enable weighted loss with --balanced here.
MobileNet is an alternative ARCH; set a new RUN path for each run to retain results.
'''), code('''ARCH = 'efficientnet_b0'  # alternative: mobilenetv3_large_100
EPOCHS = 30
BATCH_SIZE = 32  # reduce to 16 if GPU memory is insufficient
RUN = Path('outputs/full_dataset') / (ARCH + '_robust_letterbox')
assert not (RUN/'best.pt').exists(), 'Choose a new RUN path to preserve the previous checkpoint'
subprocess.run([sys.executable, '-m', 'childsafe.classifier.train',
                '--arch', ARCH, '--split', 'group', '--config', str(CONFIG),
                '--manifest', str(MANIFEST), '--robust', '--preprocessing', 'letterbox',
                '--epochs', str(EPOCHS), '--batch-size', str(BATCH_SIZE),
                '--out', str(RUN)], check=True)
metrics = json.loads((RUN/'metrics.json').read_text())
print('Held-out test:', json.dumps(metrics['test'], indent=2))
print('Test by source:', json.dumps(metrics['test_by_domain'], indent=2))
print('Calibration partition:', metrics['calibration_partition'])
'''), md('''## External Open Images test
Run once after selecting the model configuration. All 4,069 test crops remain outside
training, validation and calibration. This is classification evaluation, not webcam acceptance.
Read category confusion, unsafe recall, ECE and rejection coverage alongside accuracy.
'''), code('''subprocess.run([sys.executable, '-m', 'childsafe.classifier.evaluate',
                '--run', str(RUN), '--config', str(CONFIG), '--manifest', str(MANIFEST),
                '--ood', str(OPENIMAGES_ROOT/'test')], check=True)
report = json.loads((RUN/'eval_report.json').read_text())
print('External test:', json.dumps(report['ood'], indent=2))
'''), md('''## Download checkpoint and provenance
The archive includes best.pt, metrics, logits, training manifest, run config,
data-preparation report and external evaluation. Keep attribution manifest with shared crops.
Absolute input paths record this Kaggle session; rerun preparation with new roots elsewhere.
'''), code('''ARTIFACTS = Path('/kaggle/working/full_dataset_artifacts')
ARTIFACTS.mkdir(exist_ok=True)
shutil.copytree(RUN, ARTIFACTS/RUN.name, dirs_exist_ok=True)
for p in [CONFIG, MANIFEST, MANIFEST.with_suffix('.report.json')]:
    shutil.copy2(p, ARTIFACTS/p.name)
shutil.copy2(OPENIMAGES_ROOT/'manifest.csv', ARTIFACTS/'openimages_attribution.csv')
shutil.make_archive('/kaggle/working/full_dataset_results', 'zip', ARTIFACTS)
display(FileLink('/kaggle/working/full_dataset_results.zip'))
''')]
    for i, cell in enumerate(cells):
        cell['id'] = f'full-{i:03d}'
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
          'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    out = ROOT/'notebooks'/filename
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Wrote {out} ({len(cells)} cells)')


if __name__ == '__main__':
    build_notebook()
