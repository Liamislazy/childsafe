# ChildSafe object-near-mouth warning prototype

*Nguyên mẫu xử lý tại chỗ để hỗ trợ người chăm sóc khi vật cầm tay tiến sát vùng miệng trên ảnh; không xác nhận tiếp xúc, nuốt hoặc nghẹn.*

- **Module A**: safe/unsafe object classifier trained on ChildSUn (EfficientNet-B0, MobileNetV3, VGG19-BN baseline), with a leak-free group split, temperature scaling and Grad-CAM.
- **Runtime**: one webcam/video URL, local YOLO11 segmentation selected for the device, up to four temporary person IDs, MediaPipe on original-image person crops, and separate alert histories.
- **Rule layer**: duration window + hysteresis, visible object-mask proximity and normal / caution / danger, independently observable / degraded / unobservable. Cooldown limits notifications.
- **Evaluation tools**: false alarms per hour, event recall and latency, with a distance-only baseline and ablations. Real-camera event results remain unmeasured.

Background, scope and the literature notes are in [docs/handoff.md](docs/handoff.md). Ethics rules are in [docs/ethics.md](docs/ethics.md).

For the elevated-camera runtime, use [the tested dependency pins](requirements-runtime.txt) and [the testing guide and limitations](docs/elevated_camera.md). Local detector weights are required; startup never downloads them. The default configuration is **uncalibrated** and has limited object coverage. Pen/coin detector training remains pending.

For PIISE preparation, use the [scope, demonstration and evidence pack](docs/piise/README.md).
For the fixed-table prevention trial, use [Một phút chuẩn bị góc chơi](docs/play_area_demo.md):
`python scripts/play_area_demo.py` then open http://127.0.0.1:8002/play-area.
This separate prototype checks visible knife/scissors/fork candidates without needing a face or hand;
user checkmarks are self-reports, not a safety certificate. Real-table accuracy remains unmeasured.
`scripts/check_demo_readiness.py` inventories the actual selected local weights without opening a camera.
`scripts/prepare_fair_evidence.py` exports the strict session comparison, or explicitly reports missing data.
The finalized research report on Drive is separate and is not rewritten by these tools.

For the complete ChildSUn + Open Images classification dataset (27,883 images, 11 classes), import [the Kaggle notebook](notebooks/childsafe_expanded_kaggle.ipynb) and follow [the full-data training instructions](docs/kaggle_expanded_training.md). Open Images test stays separate; validation and calibration use different groups.

After downloading `full_dataset_results.zip`, follow [the full-model demo and camera checklist](docs/full_dataset_camera_checklist.md). Both demo launchers recognize the imported checkpoint and use the multi-person object-mask pipeline for live alerts. The browser offers per-person diagnostics, sound alerts, numeric session export and independent observer markers.

The [18 October review changes](docs/review_changes_2026_10_18.md) separate contextual hazards from dataset labels, harden identity/session handling, and add scale-aware temporal object evidence. Use the [current runtime guide](docs/product_runtime_guide.md) and [targeted trial workflow](docs/targeted_camera_trials.md). Real-camera event accuracy, calibrated thresholds and performance with actual participants remain unverified; this is not a choking detector.

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
| Web demo: live camera alerts (person, hands, mouth, object), degradation slider, session download | `python scripts/web_demo.py` then open http://127.0.0.1:8000 | nothing saved |
| B1. Normalizer stability | `run_webcam.py --no-classifier --log-features --condition ...` then `python scripts/b1_normalization.py --glob ...` | table |
| E2. Scenario videos to features | `python scripts/extract_features.py --videos data/recordings --ckpt ...` | `outputs/features/*.csv` |
| D3. Tune on dev (never on test) | `python scripts/evaluate_alerts.py grid --sessions outputs/scenarios --split dev` | `outputs/grid_dev_config.json` |
| E1/E4. Report on frozen test sessions | `python scripts/evaluate_alerts.py report --sessions outputs/scenarios --split test` | `outputs/report_test.json` |
| E1. Dev trade-off points | `python scripts/evaluate_alerts.py tradeoff --sessions outputs/scenarios --split dev` | `outputs/tradeoff_dev.json` |

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
- **Dataset labels are not contextual safety judgments.** Modern localized proximity requests caution even for toy/fruit/spoon labels. High alerts require an explicitly scoped hazard, reliable temporal mask evidence and dwell. Legacy CSVs preserve the historical policy.
- **Known limitations.** Image proximity does not establish physical contact or insertion. A box or hand near the head cannot confirm danger. Pretrained YOLO11n-seg does not locate pens/coins. Elevated thresholds and actual multi-person webcam performance remain unvalidated; this is not ready for real deployment.

## Detector bút và đũa

Luồng train vị trí vật riêng nằm trong [hướng dẫn detector](docs/pen_chopsticks_detector_training.md) và [notebook Kaggle](notebooks/childsafe_detector_kaggle.ipynb). Dữ liệu nguyên khung/bounding box khác với crop phân loại; model hai lớp này cần kiểm chứng trước khi ghép vào camera.

Lượt sửa camera tiếp theo có [runtime và quy trình train vật nguy cơ mở rộng](docs/expanded_hazard_runtime.md)
và [notebook detector nhiều lớp](notebooks/childsafe_hazard_detector_kaggle.ipynb). Danh mục gán nhãn gồm
15 loại; chỉ lớp có dữ liệu thật mới được train. Detector riêng ghép qua `--object-model`, giữ detector người.
Trọng số trong máy hiện tại vẫn chưa nhận riêng bút/đũa/đồng xu; không coi danh mục cấu hình là model đã học.
