"""Build a standalone Kaggle notebook from the colleague's original training pipeline."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}
def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':s.splitlines(keepends=True)}
files=['configs/default.yaml','scripts/prepare_expanded_data.py','data/splits/splits.csv']
files += [str(p.relative_to(ROOT)).replace('\\','/') for folder in ('src/childsafe/data','src/childsafe/classifier')
          for p in sorted((ROOT/folder).glob('*.py'))]
files += ['src/childsafe/__init__.py','src/childsafe/config.py']
cells=[md('''# ChildSafe: expanded object classifier
Adapted from the original EfficientNet/MobileNet training pipeline. No results are claimed before running.

1. Upload `childsafe_supplement.zip` as a private Kaggle Dataset, and add ChildSUn as another input.
2. The starter contains 169 reviewed crops. Recheck the sheets and edit `manifest.csv` as needed.
   New downloads start with `reviewed=false`; set true only after checking category and privacy.
   You can review locally before uploading, or export/edit/re-upload the manifest on Kaggle.
3. Enable GPU and Internet (ImageNet weights/dependencies). Set the two explicit paths below.
4. Run all. Grouped train/validation/calibration/test partitions are checked before training.
   Canonical ChildSUn test stays fixed. New source photos never cross partitions.

`unsafe` means a project policy for object-to-mouth contact, not a claim that a pen is always dangerous.
Fruit/spoon being classified safe does not establish safe eating or absence of choking.
Do not use children, sharp objects, or real insertion into anyone's mouth for scenario collection.
'''),code('''%pip install -q timm imagehash scikit-learn pandas scipy matplotlib pyyaml tqdm
import os, sys, subprocess
from pathlib import Path
REPO=Path('/kaggle/working/childsafe')
REPO.mkdir(exist_ok=True); os.chdir(REPO)
sys.path.insert(0,str(REPO/'src')); os.environ['PYTHONPATH']=str(REPO/'src')
for folder in ['configs','scripts','src/childsafe/data','src/childsafe/classifier','data/splits','outputs']:
    (REPO/folder).mkdir(parents=True,exist_ok=True)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'Enable a Kaggle GPU before the full training run'
''')]
for name in files:
    cells.append(code(f'%%writefile {name}\n'+((ROOT/name).read_text(encoding='utf-8') or '# package\n')))
cells += [md('## Explicit dataset paths — no automatic selection of the biggest image folder'),code('''# Edit these paths using the Kaggle Input panel. RAW_ROOT is the folder containing
# canonical filenames such as baby_items_10_0_1720.jpg (not the outer dataset folder).
RAW_ROOT=Path('/kaggle/input/CHANGE-ME-CHILDSUN/CHANGE-ME-IMAGE-FOLDER')
SUPPLEMENT=Path('/kaggle/input/CHANGE-ME-SUPPLEMENT/manifest.csv')
assert RAW_ROOT.is_dir(), RAW_ROOT
assert SUPPLEMENT.is_file(), SUPPLEMENT
subprocess.run([sys.executable,'scripts/prepare_expanded_data.py','--raw-root',str(RAW_ROOT),
                '--supplement',str(SUPPLEMENT)],check=True)
import pandas as pd
df=pd.read_csv('data/splits/expanded.csv')
assert {'pen','coin'}.issubset(set(df.category)), 'Approve enough pen/coin sources before training'
display(df.groupby(['category','domain','split_group']).size().unstack(fill_value=0))
'''),md('''## Training and controlled comparisons
First compare preprocessing/augmentation using validation, then choose one variant before interpreting test.
The notebook runs one chosen improved model by default; optional baseline/MobileNet runs cost extra GPU time.
Do not choose thresholds or model variants by looking at test scores repeatedly.
'''),code('''ARCH='efficientnet_b0'  # Alternative: mobilenetv3_large_100
EPOCHS=30
cmd=[sys.executable,'-m','childsafe.classifier.train','--arch',ARCH,'--split','group',
     '--manifest','data/splits/expanded.csv','--robust','--balanced','--preprocessing','letterbox',
     '--epochs',str(EPOCHS),'--batch-size','32','--out',f'outputs/expanded/{ARCH}']
subprocess.run(cmd,check=True)
'''),md('''## Inspect results, then collect deployment-domain holdouts
Read pen/coin recall and confusion with fruit/spoon, unsafe recall, ECE, rejection coverage,
confident unsafe-to-safe errors, and test metrics by source domain. Small new holdouts have wide uncertainty.
An Open Images score does not validate webcam behavior.
Add consented adult/doll hand crops with different people, rooms, devices, and physical objects;
use one `group` per session/object so neighboring frames never cross partitions.
For actual alerts, measure event recall, false alarms/hour, and detection delay on separate scenarios.
'''),code('''import json, shutil
metrics=json.loads(Path(f'outputs/expanded/{ARCH}/metrics.json').read_text())
print(json.dumps(metrics['test'],indent=2))
print('By domain:',json.dumps(metrics['test_by_domain'],indent=2))
shutil.make_archive('/kaggle/working/childsafe_expanded_results','zip','outputs/expanded')
print('Download childsafe_expanded_results.zip. The deployable checkpoint is best.pt.')
''')]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                             'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
for i,cell in enumerate(cells): cell['id']=f'expanded-{i:03d}'
out=ROOT/'notebooks/childsafe_expanded_kaggle.ipynb'; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print(out)
