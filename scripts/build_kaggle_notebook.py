"""Build notebooks/childsafe_kaggle.ipynb: a self-contained Kaggle notebook for Module A.

The notebook embeds the current project source files (%%writefile cells), so re-run this
script after changing the code:

    python scripts/build_kaggle_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "configs/default.yaml",
    "src/childsafe/__init__.py",
    "src/childsafe/config.py",
    "src/childsafe/data/__init__.py",
    "src/childsafe/data/catalog.py",
    "src/childsafe/data/dedup.py",
    "src/childsafe/data/split.py",
    "src/childsafe/data/corruptions.py",
    "src/childsafe/data/dataset.py",
    "src/childsafe/classifier/__init__.py",
    "src/childsafe/classifier/models.py",
    "src/childsafe/classifier/calibration.py",
    "src/childsafe/classifier/engine.py",
    "src/childsafe/classifier/train.py",
    "src/childsafe/classifier/evaluate.py",
    "src/childsafe/classifier/gradcam.py",
    "scripts/inspect_dataset.py",
    "scripts/make_splits.py",
    "scripts/train_all.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, collapsed=False):
    meta = {"jupyter": {"source_hidden": True}} if collapsed else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Child dangerous-object warning system — Module A on Kaggle

ChildSUn safe/unsafe classifier: leak-free **group split** vs **random split**, 3 models
(EfficientNet-B0, MobileNetV3-Large, VGG19-BN), temperature scaling, corruption curves, Grad-CAM.

**Before running**
1. *Add Input* → search **ChildSUn** (`fairuzkhan002/datasetvgg19`) → Add.
2. *Settings* → **Accelerator: GPU T4 x2** (or T4). The P100 may not be supported by the current PyTorch build.
3. *Settings* → **Internet: On** (needed for `pip install imagehash` and the pretrained ImageNet weights).
4. *Run All*. Results go to `/kaggle/working/childsafe/outputs` and `docs/`; the last cell zips them for download.

Module B (webcam + MediaPipe) cannot run on Kaggle — run it locally.
"""),
    md("## 1. Setup"),
    code("""
!pip install -q imagehash timm
import os, sys, glob, subprocess
REPO = "/kaggle/working/childsafe"
for d in ["configs", "scripts", "src/childsafe/data", "src/childsafe/classifier", "docs", "outputs", "data/splits"]:
    os.makedirs(f"{REPO}/{d}", exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, f"{REPO}/src")
os.environ["PYTHONPATH"] = f"{REPO}/src"
import torch; print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
"""),
    md("## 2. Project source (generated from the repo — expand to read)"),
]
for f in FILES:
    src = (ROOT / f).read_text(encoding="utf-8") or "# package\n"  # %%writefile rejects an empty body
    cells.append(code(f"%%writefile {f}\n{src}", collapsed=True))

cells += [
    md("## 3. Locate the dataset and write the Kaggle config"),
    code("""
imgs = glob.glob("/kaggle/input/**/*.jpg", recursive=True)
assert imgs, "No images found: add the ChildSUn dataset with 'Add Input'."
# the folder holding most images (ChildSUn ships flat: class is the filename prefix)
from collections import Counter
DATA = Counter(os.path.dirname(p) for p in imgs).most_common(1)[0][0]
print(len(imgs), "images; using", DATA)

FAST = False  # True: 8 epochs per run for a quick check; False: config default (25, early stopping)
cfg = f'''_base_: default.yaml
paths:
  raw_data: {DATA}
train:
  num_workers: 4
  batch_size: 32
'''
if FAST:
    cfg += "  epochs: 8\\n"
open("configs/kaggle.yaml", "w").write(cfg)
print(cfg)
"""),
    md("## 4. Dataset report and leak-free split (A1)"),
    code("""
!python scripts/inspect_dataset.py --config configs/kaggle.yaml
"""),
    code("""
!python scripts/make_splits.py --config configs/kaggle.yaml 2>&1 | grep -v "it/s"
"""),
    code("""
from IPython.display import Image as Show, display
for p in sorted(glob.glob("outputs/dedup/*.jpg")):
    print(p); display(Show(p, width=420))
"""),
    md("""
## 5. Train 3 models × {group, random} split

Each run: train with early stopping on val macro-F1 → fit temperature on val → evaluate on test.
About 1–2 h in total on a T4 (VGG19 is the slowest). To run a subset, edit `MODELS` / `SPLITS`.
"""),
    code("""
MODELS = "efficientnet_b0 mobilenetv3_large_100 vgg19_bn"
SPLITS = "group random"
!python scripts/train_all.py --config configs/kaggle.yaml --models {MODELS} --splits {SPLITS} 2>&1 | grep -vE "UserWarning|sched.step"
"""),
    code("""
print(open("docs/results_moduleA.md").read())
"""),
    md("## 6. Calibration plots, corruption curves (A5) and Grad-CAM — group-split runs"),
    code("""
for run in sorted(glob.glob("outputs/runs/*_group")):
    print("=" * 20, run)
    !python -m childsafe.classifier.evaluate --run {run} --config configs/kaggle.yaml --corruptions 2>&1 | grep -E "s0|s5|wrote"
    !python -m childsafe.classifier.gradcam --run {run} --config configs/kaggle.yaml --n 24
"""),
    code("""
for run in sorted(glob.glob("outputs/runs/*_group")):
    for name in ["test_reliability_raw.png", "test_reliability_calibrated.png", "degradation.png"]:
        p = f"{run}/{name}"
        if os.path.exists(p):
            print(p); display(Show(p, width=480))
"""),
    code("""
for p in sorted(glob.glob("outputs/runs/efficientnet_b0_group/gradcam/*.jpg"))[:12]:
    display(Show(p, width=224))
"""),
    md("""
## 7. Optional: A5 robustness augmentation, then out-of-domain test (A2)

For A2, upload your own webcam photos as a Kaggle dataset with one folder per class
(e.g. `knife/`, `toy/`, optionally `low_light/knife/...`) and set `OOD` to its path.
"""),
    code("""
RUN_ROBUST = False
if RUN_ROBUST:
    !python -m childsafe.classifier.train --arch efficientnet_b0 --split group --robust --config configs/kaggle.yaml
    !python -m childsafe.classifier.evaluate --run outputs/runs/efficientnet_b0_group_robust --config configs/kaggle.yaml --corruptions

OOD = ""   # e.g. "/kaggle/input/my-webcam-photos"
if OOD:
    for run in sorted(glob.glob("outputs/runs/*_group*")):
        !python -m childsafe.classifier.evaluate --run {run} --config configs/kaggle.yaml --ood {OOD}
"""),
    md("## 8. Download results\n\nThe zip includes checkpoints (`best.pt`) so you can run the webcam demo locally with them."),
    code("""
!cd /kaggle/working && zip -qr childsafe_results.zip childsafe/outputs childsafe/docs childsafe/data/splits -x "*hashes.npz" "*_random/best.pt"  # random-split checkpoints skipped (VGG19 is ~550 MB)
from IPython.display import FileLink
FileLink("childsafe_results.zip")
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "childsafe_kaggle.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
