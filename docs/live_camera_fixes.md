# Live camera: speed and sensitivity fixes (2026-10-09)

Review of s4dz's elevated-camera runtime (`6ded652`). Measured on the RTX 3050 laptop with synthetic,
non-human frames (no people filmed) with `scripts/profile_live_runtime.py`, plus regression tests in
`tests/test_live_speed.py`. These are runtime measurements, not event recall or false alarms per hour:
those still need the dev sessions in `full_dataset_camera_checklist.md`.

## Why speed decided sensitivity

`AlertProcessor` resets smoothing, the evidence window, dwell and the cached classification whenever
two frames are more than `temporal.max_gap_s` (0.25 s) apart. DANGER needs 0.1 s of window evidence and
0.5 s of dwell, so any frame loop slower than 0.25 s can never raise DANGER. The rule was kept (it stops
evidence from being bridged over unobserved time); the pipeline was made fast enough instead.

## Problems found and fixes

| # | Problem | Fix | Effect |
|---|---|---|---|
| 1 | Overlapping people are marked ambiguous and get a new ID every frame; each new ID built a new MediaPipe face+hand landmarker (~0.3-0.4 s) | Pool of landmarker instances, only people tracked in the current frame hold one | 2 overlapping people: **1225 ms -> 27 ms per frame** |
| 2 | Mask-to-mouth distance computed up to 4 times per object per frame, over every mask pixel | Measured once; only pixels inside the mouth box (inside test) and mask boundary pixels (distance) are used, same result within <1 px | 2.0 ms -> 0.2 ms |
| 3 | Brightness/sharpness check on full frames in NumPy | Same formula with OpenCV | 25 -> 15 ms (720p), 12 -> 7 ms (480p) |
| 4 | Classifier ran on every frame, ignoring `cascade.classify_interval_s` | Reuse a classification of the same object less than 0.1 s old while the processor still holds it | fewer classifier calls at high FPS |
| 5 | Full-resolution masks copied to CPU for every person (unused) | Masks only for objects; device chosen explicitly (`detector.device: auto`) | less GPU->CPU copying |
| 6 | Web warning only compared server time with 0.25 s, not the real frame spacing (server + browser round trip) | Server reports `frame_gap_ms`; warning and stats line use it, shown red when over the limit | the cause of "no DANGER" is visible |
| 7 | One missed detection of a thin object restarted the 0.5 s dwell | Dwell pauses for up to `temporal.dropout_grace_s` (0.3 s) and resumes; missing time is never added, shown dwell stays 0 during the gap | flicker case: DANGER at 0.8 s instead of 1.0 s |
| 8 | Selected object switching was decided by box IoU < 0.2, so a thin object moving toward the mouth reset all history every frame | Same object if IoU >= 0.2, or same label and centre moved less than the box size | moving knife: DANGER at 0.6 s instead of never |
| 9 | Objects below 0.5 confidence were dropped, so most forks/knives never reached the CAUTION candidate path | `detector.object_min_conf: 0.35`, `quality.min_candidate_mask_conf: 0.35` (people stay 0.5) | more CAUTION for thin objects; **DANGER thresholds unchanged** (mask 0.6, P(unsafe) 0.6, dwell 0.5 s) |
| 10 | On this laptop the live model fell back to "standard" (ChildSUn-only, calls ~73% of pens unsafe) | Models with an `other` class are tried first | fewer pen/phone false alarms locally |

Items 7-9 change behaviour and are **untuned starting values**: measure false alarms per hour and
event recall on dev sessions before relying on them, and keep test sessions untouched.

## Measured components (GPU, 640x480, median)

| Component | Time |
|---|---|
| YOLO11n-seg detect | 13.6 ms (CPU: ~72 ms) |
| MediaPipe face+hand detect | 12.7 ms |
| MediaPipe landmarker creation | 70-420 ms (now once per pooled instance) |
| EfficientNet-B0 classifier | 19.8 ms (CPU: ~36 ms per s4dz's benchmark) |
| Pipeline, 1 person (no face in synthetic frame) | 14 ms |
| Pipeline, 2 overlapping people | 27 ms (was 1225 ms) |
| Web server per frame, empty scene | 17-24 ms after GPU warm-up |
