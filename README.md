# ChildSafe: on-device warning when a held object nears a child's mouth

A research prototype that watches one camera feed on a local machine and warns a caregiver when a hand-held object stays close to a person's mouth in the image. All processing happens on the device. No video is saved.

> **Scope.** ChildSafe is **not a choking detector**. It cannot confirm contact, insertion or swallowing, and it does not replace adult supervision. Its accuracy on real cameras with real participants has **not been measured yet**. Every result below is offline, on public image datasets.

## Why

Young children put small objects in their mouths, and many of these objects are choking or injury hazards. A caregiver cannot watch every second. The idea is a cheap, private early warning: notice the *approach* (a risky object moving toward the mouth and staying there) a few seconds earlier, and leave the decision to the adult.

## System overview

```text
camera frame
   │
   ├─► YOLO11-seg ── person and object masks (up to 4 temporary person IDs)
   │
   ├─► MediaPipe (on each person crop) ── hands and mouth landmarks
   │
   ├─► Module A classifier ── object crop → 10 classes → calibrated P(unsafe)
   │
   └─► rule layer ── mask-to-mouth proximity × dwell time × hysteresis
                       → normal / caution / danger  (+ observable / degraded / unobservable)
                       → cooldown-limited alert
```

- **Module A** classifies object crops from the ChildSUn dataset (baby items, ball, fork, fruit, heater, knife, scissors, socket, spoon, toy) and outputs a calibrated probability that the object is unsafe.
- **Runtime**: webcam or video URL, a local YOLO11 segmentation model chosen for the hardware, MediaPipe landmarks, a separate alert history per person.
- **Rule layer**: an alert needs visible object-mask proximity to the mouth, held for a duration window, with hysteresis. When the mouth is not visible the system says so instead of guessing. Lost mouth landmarks keep recent risk for at most 0.8 s.
- **Companion app (Android)**: pairs with the monitor, can run camera inference on the phone, and relays alerts.
- **Play-area check**: a separate demo that scans a table for visible knives, scissors or forks before play begins.
- **Evaluation tools**: false alarms per hour, event recall and latency, a distance-only baseline and ablations. These tools are built, but no real-camera sessions have been scored with them yet.

## Results (offline)

### 1. A leak-free split changes the story

ChildSUn contains augmented copies of the same source photo. With a random split, **91.8 % of test images have a copy of the same photo in the training set**. We grouped near-duplicates with perceptual hashing (pHash/dHash) and split by group.

| Model | Random split acc. | **Group split acc.** | Group macro-F1 | Group unsafe recall |
|---|---|---|---|---|
| EfficientNet-B0 | 0.993 | **0.920** | 0.896 | 0.987 |
| MobileNetV3-Large | 0.998 | **0.878** | 0.855 | 0.971 |
| VGG19-BN | 0.996 | **0.915** | 0.862 | 0.961 |

The random split inflates accuracy by 7–12 points. The group split is the honest number. Each model was trained once (seed 42), so differences of 1–3 points between models should not be used to rank them.

Temperature scaling lowers the calibration error of P(unsafe), the value the alert layer uses, for all three models (EfficientNet-B0 binary ECE: 0.031 → 0.022). Full tables, per-class recall and corruption curves are in [docs/results_moduleA.md](docs/results_moduleA.md).

### 2. Out-of-domain objects

A classifier trained only on ChildSUn has never seen a pen, phone or book, so it has to force them into one of its 10 classes. We tested it on 4,069 crops from the **Open Images V7 test split**, which was never used for training.

| | ChildSUn only | + Open Images, 102 classes as "other" |
|---|---|---|
| ChildSUn accuracy | 92.1 % | 87.3 % |
| Open Images accuracy (shared classes) | 74.0 % | **90.6 %** |
| "other" objects called unsafe | 19.5 % | **0.8 %** |
| pens called unsafe | 72.8 % | 10.9 % |

Adding an "other" class trades about 5 points on ChildSUn for much better behaviour on everyday objects. See [docs/results_openimages.md](docs/results_openimages.md).

### Known weaknesses

- Gaussian and ISO noise drop accuracy sharply (EfficientNet-B0: 0.92 → 0.15 at severity 3).
- Spoon has only 5 test images, so its recall is not reliable.
- The pretrained YOLO11n-seg detector does not locate pens or coins. A dedicated detector is still in training.
- Default thresholds are uncalibrated. Proximity in the image does not prove physical contact.

## Ethics and privacy

- No video of real children was recorded. Tests use adults and dolls, with consent for every adult recording.
- Processing is on-device. Event logs store numbers only, never frames.
- The project follows Vietnam's Personal Data Protection Law (91/2025/QH15).

Details: [docs/ethics.md](docs/ethics.md).

## Getting started

Windows with a local NVIDIA GPU, Python 3.12 (MediaPipe has no wheels for 3.14):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv -e .
.venv\Scripts\python scripts\download_mediapipe_models.py
```

Web demo (live alerts, per-person diagnostics, degradation slider, numeric session export):

```bash
python scripts/web_demo.py        # then open http://127.0.0.1:8000
```

Detector weights must already be on the machine. Startup never downloads them.

The full training and evaluation pipeline (dataset download, deduplicated splits, training, calibration, Grad-CAM, OOD and corruption tests, alert tuning on dev sessions and reporting on frozen test sessions) is documented step by step in **[docs/developer_guide.md](docs/developer_guide.md)**. All thresholds live in [configs/default.yaml](configs/default.yaml).

## Repository layout

```text
configs/                  thresholds and paths
src/childsafe/data/       catalog, perceptual-hash dedup, group split, corruptions
src/childsafe/classifier/ models, training, calibration, evaluation, Grad-CAM
src/childsafe/vision/     detection, landmarks, geometry, smoothing
src/childsafe/alerts/     temporal rules, alert processor, privacy-preserving event log
src/childsafe/evaluation/ event metrics, replay of logged features
src/childsafe/runtime/    multi-person live pipeline
src/childsafe/web/        browser demo
android/                  companion app
notebooks/                Kaggle training notebooks
docs/                     results, ethics, guides, PIISE evidence pack
tests/                    pytest
```

## Team

| | Main contributions |
|---|---|
| **Cao Tung Lam** ([@Liamislazy](https://github.com/Liamislazy)) | repository owner; Module A training and evaluation (ChildSUn, Open Images), detector selection (YOLO11), object-to-mouth measurement, mouth observability, knife-in-hand voting |
| **Phan Xuân Khoa** ([@s4dz](https://github.com/s4dz)) | most of the multi-person live runtime and browser demo, Android companion (pairing, on-device camera inference, relay), play-area check, PIISE demonstration and evidence pack |

The alert rules and vision code were written by both. The commit history has the full record.

Background, scope and literature notes: [docs/handoff.md](docs/handoff.md). PIISE materials: [docs/piise/README.md](docs/piise/README.md).
