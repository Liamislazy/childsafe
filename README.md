# Child dangerous-object warning system

*Hệ thống cảnh báo đồ vật nguy hiểm cho trẻ nhỏ dựa trên thị giác máy tính: kết hợp phân loại đồ vật an toàn/không an toàn và giám sát khoảng cách tới miệng theo thời gian thực.*

- **Module A**: safe/unsafe object classifier trained on ChildSUn (EfficientNet-B0, MobileNetV3, VGG19-BN baseline), with a leak-free group split, temperature scaling and Grad-CAM.
- **Module B**: one webcam + MediaPipe Face/Hand Landmarker. Hand-to-mouth distance is normalized by face width (a **relative** distance), then smoothed.
- **Rule layer**: N-of-M + hysteresis temporal rule, cascade gating and an alert matrix, giving normal / caution / danger.
- **Evaluation**: false alarms per hour, event recall, latency, compared with a per-frame distance-only baseline, plus ablations.

Background, scope and the literature notes are in [docs/handoff.md](docs/handoff.md). Ethics rules are in [docs/ethics.md](docs/ethics.md).

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
| Web demo: two models side by side, degradation slider, photo test | `python scripts/web_demo.py` then open http://127.0.0.1:8000 | nothing saved |
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
- **Offline extraction classifies every hand frame.** The cascade (D2) is then applied during replay, so it can be switched off in the ablation.
- **Lost mouth landmark counts as a signal (B2).** If the face disappears while the hand was near it, or while the hand stays at the last known mouth position, the level goes to caution.
- **Safe objects cap at caution.** Low calibrated confidence becomes "uncertain", which also gives caution.
- **Known limitations.** The distance is a 2D relative measure (depth ambiguity). The classifier sees a hand crop, and ChildSUn has no "empty hand" class. The landmark models were trained on adults.
