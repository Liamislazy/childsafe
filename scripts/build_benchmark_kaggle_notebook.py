"""Build notebooks/childsafe_benchmark_kaggle.ipynb: train extra Module A architectures on Kaggle with the
same recipe and canonical group split as the original three (configs/default.yaml, data/splits/splits.csv,
checksum 42e3fb2547c5), so every model in the benchmark is trained on the same hardware.
Speed and robustness are measured afterwards on the target laptop with scripts/benchmark_classifiers.py.
"""
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (timm architecture with ImageNet-1k weights, seed). EfficientNet-B0 seed 42 already exists from Kaggle v3;
# seeds 1 and 2 measure run-to-run spread of the chosen model.
RUNS = [('mobilenetv3_small_100.lamb_in1k', 42), ('resnet18.a1_in1k', 42), ('resnet50.a1_in1k', 42),
        ('convnext_tiny.fb_in1k', 42), ('efficientnet_b0.ra_in1k', 1), ('efficientnet_b0.ra_in1k', 2)]


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': s.splitlines(keepends=True)}


def build_notebook(filename='childsafe_benchmark_kaggle.ipynb'):
    files = ['configs/default.yaml']
    files += [p.relative_to(ROOT).as_posix() for folder in ('src/childsafe/data', 'src/childsafe/classifier')
              for p in sorted((ROOT/folder).glob('*.py'))]
    files += ['src/childsafe/__init__.py', 'src/childsafe/config.py']
    cells = [md('''# ChildSafe: benchmark thêm kiến trúc phân loại trên Kaggle
Huấn luyện thêm MobileNetV3-Small, ResNet-18, ResNet-50, ConvNeXt-Tiny và EfficientNet-B0 với seed 1, 2,
cùng công thức (configs/default.yaml) và cùng phép chia theo nhóm (data/splits/splits.csv, checksum 42e3fb2547c5)
như ba mô hình cũ. Chỉ dùng ChildSUn. Tốc độ và độ bền được đo sau trên laptop bằng scripts/benchmark_classifiers.py.

1. Add Input: dataset ChildSUn `fairuzkhan002/datasetvgg19`.
2. Settings: GPU T4, Internet bật. Thư mục ảnh được tìm tự động.
3. Run All. Khoảng 40–60 phút. Cell huấn luyện bỏ qua lượt đã xong nên có thể chạy lại nếu bị ngắt.
4. Tải `benchmark_results.zip` ở cuối.
'''), code('''%pip install -q timm scikit-learn pandas scipy matplotlib pyyaml pillow opencv-python-headless
import os, sys, subprocess, json, shutil, hashlib
from pathlib import Path
import pandas as pd
import yaml
from IPython.display import display, FileLink

REPO = Path('/kaggle/working/childsafe')
REPO.mkdir(exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO/'src'))
os.environ['PYTHONPATH'] = str(REPO/'src')
for folder in ['configs', 'src/childsafe/data', 'src/childsafe/classifier', 'data/splits', 'outputs/benchmark/runs']:
    (REPO/folder).mkdir(parents=True, exist_ok=True)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'Enable the GPU accelerator first'
'''), md('## Mã nguồn và phép chia cố định (nhúng sẵn)')]
    payload = base64.b64encode(gzip.compress((ROOT/'data/splits/splits.csv').read_bytes(), mtime=0)).decode('ascii')
    cells.append(code("import base64, gzip\n" +
                      f"Path('data/splits/splits.csv').write_bytes(gzip.decompress(base64.b64decode({payload!r})))\n"))
    for name in files:
        cells.append(code(f'%%writefile {name}\n' + ((ROOT/name).read_text(encoding='utf-8') or '# package\n')))
    cells += [md('## Đường dẫn dữ liệu và kiểm tra phép chia'), code('''# Folder of the attached ChildSUn input that holds the canonical filenames (found automatically).
hits = sorted({p.parent for p in Path('/kaggle/input').rglob('baby_items_10_0_1720.jpg')})
assert len(hits) == 1, f'Attach the ChildSUn dataset exactly once: {hits}'
RAW_ROOT = hits[0]
print('RAW_ROOT =', RAW_ROOT)
splits = pd.read_csv('data/splits/splits.csv')
missing = [p for p in splits.path if not (RAW_ROOT/p).is_file()]
assert not missing, f'{len(missing)} images missing, e.g. {missing[:3]}'
checksum = hashlib.md5(''.join(f'{p}:{g}' for p, g in sorted(zip(splits.path, splits.split_group))).encode()).hexdigest()[:12]
assert checksum == '42e3fb2547c5', checksum
CONFIG = Path('configs/kaggle_benchmark.yaml')
CONFIG.write_text(yaml.safe_dump({'_base_': 'default.yaml',
    'paths': {'raw_data': str(RAW_ROOT), 'outputs': 'outputs/benchmark'},
    'train': {'num_workers': 2}}, sort_keys=False), encoding='utf-8')
display(splits.groupby(['category', 'split_group']).size().unstack(fill_value=0))
'''), md('''## Huấn luyện
Mỗi lượt dùng `childsafe.classifier.train` như các mô hình cũ: AdamW, OneCycle, lr 3e-4, batch 32, tối đa 25 epoch,
dừng sớm sau 6 epoch, chọn checkpoint theo macro-F1 trên val, temperature scaling trên val. Không có lựa chọn
nào dựa trên tập test.
'''), code(f'''RUNS = {RUNS!r}
summary = []
for arch, seed in RUNS:
    name = arch.split('.')[0] + ('' if seed == 42 else f'_seed{{seed}}') + '_group'
    out = Path('outputs/benchmark/runs')/name
    if not (out/'metrics.json').exists():
        subprocess.run([sys.executable, '-m', 'childsafe.classifier.train', '--arch', arch, '--split', 'group',
                        '--seed', str(seed), '--config', str(CONFIG), '--out', str(out)], check=True)
    t = json.loads((out/'metrics.json').read_text())['test']
    summary.append({{'run': name, 'accuracy': round(t['accuracy'], 4), 'macro_f1': round(t['macro_f1'], 4),
                    'unsafe_recall': round(t['binary']['unsafe_recall'], 4)}})
display(pd.DataFrame(summary))
'''), md('## Tải kết quả'), code('''shutil.make_archive('/kaggle/working/benchmark_results', 'zip', 'outputs/benchmark')
display(FileLink('/kaggle/working/benchmark_results.zip'))
''')]
    for i, cell in enumerate(cells):
        cell['id'] = f'bench-{i:03d}'
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
          'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    out = ROOT/'notebooks'/filename
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Wrote {out} ({len(cells)} cells)')


if __name__ == '__main__':
    build_notebook()
