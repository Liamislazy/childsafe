# Child dangerous-object warning system

*Hệ thống cảnh báo đồ vật nguy hiểm cho trẻ nhỏ dựa trên thị giác máy tính: kết hợp phân loại đồ vật an toàn/không an toàn và giám sát khoảng cách tới miệng theo thời gian thực.*

- **Module A**: safe/unsafe object classifier trained on ChildSUn (EfficientNet-B0, MobileNetV3, VGG19-BN baseline), with a leak-free group split, temperature scaling and Grad-CAM.
- **Runtime**: one webcam/video URL, local YOLO11n segmentation, up to four temporary person IDs, MediaPipe on original-image person crops, and separate alert histories.
- **Rule layer**: duration window + hysteresis, visible object-mask proximity and normal / caution / danger, independently observable / degraded / unobservable. Cooldown limits notifications.
- **Evaluation**: false alarms per hour, event recall, latency, compared with a per-frame distance-only baseline, plus ablations.

Background, scope and the literature notes are in [docs/handoff.md](docs/handoff.md). Ethics rules are in [docs/ethics.md](docs/ethics.md).

For the elevated-camera runtime, use [the tested dependency pins](requirements-runtime.txt) and [the testing guide and limitations](docs/elevated_camera.md). Local detector weights are required; startup never downloads them. The default configuration is **uncalibrated** and has limited object coverage. Large dataset downloads and pen/coin detector training remain paused.

## Setup (Windows, local GPU)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv -e .
.venv\Scripts\python scripts\download_mediapipe_models.py
```

Use Python 3.12, because MediaPipe has no wheels for 3.14. On Colab, run `pip install -r requirements.txt && pip install -e .`.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Download ChildSUn (needs `~/.kaggle/kaggle.json`, or download by hand into `data/raw/childsun`) | `python scripts/download_childsun.py` | `data/raw/childsun` |
| 2. Inspect: folders, counts, license, augmentation hints | `python scripts/inspect_dataset.py` | `docs/dataset_report.md` |
| 3. A1: pHash/dHash near-duplicate groups; group split and random split | `python scripts/make_splits.py` | `data/splits/splits.csv`, `docs/split_report.md`, montages |
| 4-6. Train 3 models on both splits; calibrate on val; test | `python scripts/train_all.py` | `outputs/runs/*`, `docs/results_moduleA.md` |
| A2/A5. OOD webcam photos and degradation curves | `python -m childsafe.classifier.evaluate --run outputs/runs/efficientnet_b0_group --ood data/ood --corruptions` | `eval_report.json`, `degradation.png` |
| A5. Robust augmentation | `python -m childsafe.classifier.train --arch efficientnet_b0 --split group --robust` | |
| Grad-CAM | `python -m childsafe.classifier.gradcam --run outputs/runs/efficientnet_b0_group` | `gradcam/*.jpg` |
| Live demo | `python scripts/run_webcam.py --ckpt outputs/runs/efficientnet_b0_group/best.pt` | `outputs/events/*.csv` (no video) |
| B1. Normalizer stability | `run_webcam.py --no-classifier --log-features --condition ...` then `python scripts/b1_normalization.py --glob ...` | table |
| E2. Scenario videos to features | `python scripts/extract_features.py --videos data/recordings --ckpt ...` | `outputs/features/*.csv` |
| D3. Tune on dev (never on test) | `python scripts/evaluate_alerts.py grid --split dev` | `outputs/tuned_config.yaml` |
| E1/E4. Report on test | `python scripts/evaluate_alerts.py report --split test --config outputs/tuned_config.yaml` | `docs/results_alerts_test.md` |
| E1. Trade-off curve | `python scripts/evaluate_alerts.py tradeoff --split test` | `outputs/tradeoff_test.png` |

Annotation format: [data/scenarios/README.md](data/scenarios/README.md). Every threshold is in [configs/default.yaml](configs/default.yaml).

## Layout

```
configs/default.yaml        all thresholds and paths
src/childsafe/data/         catalog, perceptual-hash dedup, group split, corruptions, dataset
src/childsafe/classifier/   timm models, train, calibration, evaluate, Grad-CAM
src/childsafe/vision/       MediaPipe landmarks, geometry/normalization, smoothing, hand-crop classifier
src/childsafe/alerts/       temporal rules, AlertProcessor (D1-D3), privacy-preserving event log
src/childsafe/evaluation/   event metrics (E1), replay of logged features
src/childsafe/pipeline.py   frame -> landmarks -> classifier -> alert (shared by live and offline)
scripts/                    command-line entry points
tests/                      pytest (python -m pytest)
```

## Design notes

- **Live and offline use the same code.** `AlertProcessor` runs both in the webcam app and on logged feature CSVs. Tuning (D3) and ablation (E4, `Components`) therefore run on numbers only, without re-processing video.
- **Offline extraction classifies detected object crops.** Schema v2 logs geometry and quality per person; replay consumes the same observations. Legacy CSVs retain a single-person compatibility mode with unknown observation coverage. Frame/hand classification options from older scripts do not fabricate object positions.
- **Lost mouth landmarks preserve recent risk for at most 0.8 seconds.** Held coordinates never accumulate proximity dwell. Events end with explicit departure, observation-loss, person-exit or disconnect reasons.
- **Safe objects cap at caution.** Low calibrated confidence becomes "uncertain", which also gives caution.
- **Known limitations.** Image proximity does not establish physical contact or insertion. A box or hand near the head cannot confirm danger. Pretrained YOLO11n-seg does not locate pens/coins. Elevated thresholds and actual multi-person webcam performance remain unvalidated; this is not ready for real deployment.
