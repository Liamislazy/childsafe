"""Build notebooks/childsafe_benchmark_eval_kaggle.ipynb: evaluate every Module A benchmark checkpoint on one Kaggle
machine (accuracy from metrics.json, fixed-seed corruptions, Open Images test crops, size, GPU and CPU latency) with
scripts/benchmark_classifiers.py, and draw the accuracy-vs-CPU-latency figure.

Kaggle inputs: ChildSUn (fairuzkhan002/datasetvgg19), the private dataset liamislazy/childsafe-benchmark-inputs
(four original Kaggle v3 checkpoints + Open Images test crops) and the output of the training notebook
liamislazy/notebookc7aa0221a6 (six new checkpoints).
"""
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# run folder -> display name (order = table order)
MODELS = {
    'efficientnet_b0_group': 'EfficientNet-B0',
    'efficientnet_b0_seed1_group': 'EfficientNet-B0 (seed 1)',
    'efficientnet_b0_seed2_group': 'EfficientNet-B0 (seed 2)',
    'efficientnet_b0_group_robust': 'EfficientNet-B0 bền',
    'mobilenetv3_large_100_group': 'MobileNetV3-Large',
    'mobilenetv3_small_100_group': 'MobileNetV3-Small',
    'resnet18_group': 'ResNet-18',
    'resnet50_group': 'ResNet-50',
    'convnext_tiny_group': 'ConvNeXt-Tiny',
    'vgg19_bn_group': 'VGG19-BN',
}


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': s.splitlines(keepends=True)}


def build_notebook(filename='childsafe_benchmark_eval_kaggle.ipynb'):
    files = ['configs/default.yaml', 'scripts/benchmark_classifiers.py', 'scripts/compare_ood.py',
             'scripts/plot_classifier_benchmark.py']
    files += [p.relative_to(ROOT).as_posix() for folder in ('src/childsafe/data', 'src/childsafe/classifier')
              for p in sorted((ROOT/folder).glob('*.py'))]
    files += ['src/childsafe/__init__.py', 'src/childsafe/config.py']
    cells = [md('''# ChildSafe: benchmark bộ phân loại trên một máy Kaggle
Đo tất cả checkpoint trên cùng phần cứng: độ chính xác (metrics.json), biến dạng ảnh seed cố định, ảnh Open Images
chưa từng huấn luyện, số tham số, thời gian một ảnh trên GPU, CPU nhiều luồng và CPU 1 luồng. Không chọn mô hình
hay ngưỡng nào dựa trên tập test.
'''), code('''%pip install -q timm scikit-learn pandas scipy matplotlib pyyaml pillow opencv-python-headless
import os, sys, subprocess, json, zipfile, hashlib
from pathlib import Path
import pandas as pd
import yaml
REPO = Path('/kaggle/working/childsafe')
for folder in ['configs', 'scripts', 'src/childsafe/data', 'src/childsafe/classifier', 'data/splits', 'data/external']:
    (REPO/folder).mkdir(parents=True, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO/'src'))
os.environ['PYTHONPATH'] = str(REPO/'src')
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print([l for l in open('/proc/cpuinfo') if l.startswith('model name')][:1], os.cpu_count(), 'CPUs')
''')]
    payload = base64.b64encode(gzip.compress((ROOT/'data/splits/splits.csv').read_bytes(), mtime=0)).decode('ascii')
    cells.append(code("import base64, gzip\n" +
                      f"Path('data/splits/splits.csv').write_bytes(gzip.decompress(base64.b64decode({payload!r})))\n"))
    for name in files:
        cells.append(code(f'%%writefile {name}\n' + ((ROOT/name).read_text(encoding='utf-8') or '# package\n')))
    cells += [md('## Tìm dữ liệu và checkpoint'), code(f'''UNPACK = Path('/kaggle/working/unpacked')
UNPACK.mkdir(exist_ok=True)
for z in sorted(Path('/kaggle/input').rglob('*.zip')):
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        if any(n.endswith('best.pt') or n.startswith('openimages/') or n.endswith('openimages/manifest.csv') for n in names):
            f.extractall(UNPACK/z.stem)
            print('unpacked', z)
hits = sorted({{p.parent for p in Path('/kaggle/input').rglob('baby_items_10_0_1720.jpg')}})
assert len(hits) == 1, hits
RAW_ROOT = hits[0]
splits = pd.read_csv('data/splits/splits.csv')
checksum = hashlib.md5(''.join(f'{{p}}:{{g}}' for p, g in sorted(zip(splits.path, splits.split_group))).encode()).hexdigest()[:12]
assert checksum == '42e3fb2547c5', checksum
oi = sorted({{p.parent for root in (Path('/kaggle/input'), UNPACK) for p in root.rglob('manifest.csv') if (p.parent/'test').is_dir()}})
assert oi, 'Open Images test crops not found'
link = REPO/'data/external/openimages'
if not link.exists():
    link.symlink_to(oi[0], target_is_directory=True)
CONFIG = Path('configs/kaggle_eval.yaml')
CONFIG.write_text(yaml.safe_dump({{'_base_': 'default.yaml', 'paths': {{'raw_data': str(RAW_ROOT)}}}}), encoding='utf-8')
MODELS = {MODELS!r}
found = {{}}
for root in (Path('/kaggle/input'), UNPACK):
    for p in root.rglob('best.pt'):
        if p.parent.name in MODELS and (p.parent/'metrics.json').exists():
            found.setdefault(p.parent.name, p)
missing = [m for m in MODELS if m not in found]
assert not missing, missing
for k, p in found.items():
    print(f'{{MODELS[k]:28s}} {{p}}')
'''), md('## Benchmark'), code('''cmd = [sys.executable, 'scripts/benchmark_classifiers.py', '--config', str(CONFIG), '--out', '/kaggle/working/benchmark.json']
for k, name in MODELS.items():
    cmd += ['--model', f'{name}={found[k]}']
subprocess.run(cmd, check=True)
res = json.loads(Path('/kaggle/working/benchmark.json').read_text())
cols = ['name', 'params_m', 'accuracy', 'clean_check_accuracy', 'unsafe_recall', 'mean_corruption_acc_s3',
        'oi_acc_shared', 'latency_gpu_ms', 'latency_cpu_ms', 'latency_cpu_1thread_ms']
display(pd.DataFrame(res['models'])[cols].round(4))
print(res['machine'])
'''), md('## Hình 2'), code('''subprocess.run([sys.executable, 'scripts/plot_classifier_benchmark.py', '/kaggle/working/hinh2_benchmark.png',
                '/kaggle/working/benchmark.json'], check=True)
from IPython.display import Image, display
display(Image('/kaggle/working/hinh2_benchmark.png'))
''')]
    for i, cell in enumerate(cells):
        cell['id'] = f'eval-{i:03d}'
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
          'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    out = ROOT/'notebooks'/filename
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Wrote {out} ({len(cells)} cells)')
    return out


if __name__ == '__main__':
    build_notebook()
