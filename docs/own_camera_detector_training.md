# Fine-tuning the detector on your own webcam

Goal: make `yolo11n-seg` better on the low-quality demo webcam by training on frames from that webcam.
Script: `scripts/own_camera_detector.py` (capture → prelabel → review → build → train).

**Who may be recorded:** you and other consenting adult volunteers, and dolls. No real children
(docs/ethics.md). Props stay blunt, large and non-swallowable; stop **before** the mouth, never put
anything in the mouth. Frames go to `data/recordings/detector_frames/` (git-ignored), never to git,
Kaggle public datasets, or a cloud API. Delete them after the project.

## 1. Plan sessions

A session is one recording run. Each session belongs to exactly one split, chosen when you start
it. Example plan (about 10 sessions, about 2-4 min each):

| split | sessions | conditions to cover |
|---|---|---|
| train | 6 | normal light, dim, backlight (window behind you), fast motion, near and far, doll in hand |
| val | 2 | mix of the same conditions, different day or room arrangement |
| test | 2 | mix of the same conditions, recorded last, never looked at while tuning |

In every session also record **negatives**: empty hand near the face, scratching the face, drinking
from a cup, holding a phone. Without them the model learns "anything near the face is an object".

If you are the only person, train and test share a participant. `build` reports this. Then the
test answers "same person, new session", not "new people". Say so in the report.

## 2. Capture (frames only, never video)

```powershell
.venv\Scripts\python scripts/own_camera_detector.py capture --session tr01 --split train --participant P01 --condition dim backlight --objects plastic_fork teddy_bear cup --consent-confirmed --width 1280 --height 720
```

SPACE starts/pauses saving (one frame every 0.5 s), Q quits. Use the **same camera and resolution
as the demo**. Brightness/sharpness per frame go to `frames.json`.

## 3. Pseudo-labels from the larger model

```powershell
.venv\Scripts\python scripts/own_camera_detector.py prelabel data/recordings/detector_frames/tr01 --teacher models/yolo11m-seg.pt
```

`yolo11m-seg` at 960 px finds more than `n` at 640 px. Its labels are still guesses: status
`unreviewed_pseudo_labels`.

## 4. Review

```powershell
.venv\Scripts\python scripts/own_camera_detector.py review data/recordings/detector_frames/tr01
```

Keys: `a` correct (copied to `labels_reviewed/`), `f` needs fixing, `x` exclude (unusable frame), `q` stop
(resume later). Be strict. A missed fork or a box on your hand labelled "knife" is `f`.

Fix the `f` frames in a polygon tool, e.g. CVAT, which has YOLO segmentation import/export
**[UNVERIFIED: check the format name in your CVAT version]**. Save each fixed `.txt` into
`labels_reviewed/` with the same file name. Do not exclude `f` frames because they are hard. The
hard frames are the reason for this work, and dropping them makes the test score look better than it is.

## 5. Build the dataset

```powershell
.venv\Scripts\python scripts/own_camera_detector.py build data/recordings/detector_frames/tr01 data/recordings/detector_frames/tr02 data/recordings/detector_frames/va01 data/recordings/detector_frames/te01 --out data/detector/own_camera_v1 --allow-unreviewed-train
```

- val and test use **reviewed labels only**. `--allow-unreviewed-train` lets unreviewed teacher labels
  into train only (distillation from m to n); frames marked `f` are never used until fixed.
- It refuses missing consent, identical frames across splits, invalid rows, and empty splits.
- `report.json` counts reviewed and pseudo labels per split. Put those counts in the report.

## 6. Train and compare

```powershell
.venv\Scripts\python scripts/own_camera_detector.py train --data data/detector/own_camera_v1/dataset.yaml --output outputs/own_camera_detector_v1 --epochs 30 --device cpu
```

Use `--device 0` on a GPU. The defaults try to limit overfitting and forgetting on a small set
(lr0 0.001, first 10 layers frozen, early stopping on val). They are untuned starting values.
`evaluation.json` holds the **base and fine-tuned model on the same test sessions**. Report both,
even if fine-tuning does not help. Other COCO classes are not re-measured, so forgetting is possible.

Do not tune epochs or lr on the test split. Change them only using val results.

## 7. Use it in the app

Copy `configs/default.yaml` to a new config and set `detector.weights` to the new `best.pt`. Also set
`weights_gpu` to it, or remove `weights_gpu`, because on a CUDA machine `weights_gpu` (yolo11s-seg)
otherwise wins. Add the checkpoint SHA-256 to `docs/model_provenance.json`. Then compare alerts on
**dev** scenario sessions with the old and the new detector:
`evaluate_alerts.py grid --sessions outputs/scenarios --split dev`. The detector mAP above is not the
alert metric. False alarms per hour and missed events are.

Limitation for the report: adult/doll frames from one webcam. This does not show transfer to young
children or other cameras.
