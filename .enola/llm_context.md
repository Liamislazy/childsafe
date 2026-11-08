# Architecture Snapshot

## Repository Map

| Module | Language | Symbols | Exported |
|--------|----------|---------|----------|
| `.venv-notify/Scripts` | python | 0 | 0 |
| `android/app/src/main/kotlin/vn/childsafe/app` | kotlin | 110 | 69 |
| `android/app/src/test/kotlin/vn/childsafe/app` | kotlin | 56 | 44 |
| `android/core/src/main/kotlin/vn/childsafe/core` | kotlin | 58 | 53 |
| `android/core/src/test/kotlin/vn/childsafe/core` | kotlin | 36 | 35 |
| `scripts` | python | 107 | 105 |
| `services/notify` | python | 20 | 16 |
| `src/childsafe` | python | 32 | 15 |
| `src/childsafe/alerts` | python | 34 | 22 |
| `src/childsafe/classifier` | python | 26 | 19 |
| `src/childsafe/companion` | python | 44 | 38 |
| `src/childsafe/data` | python | 36 | 21 |
| `src/childsafe/evaluation` | python | 34 | 29 |
| `src/childsafe/runtime` | python | 44 | 33 |
| `src/childsafe/vision` | python | 82 | 56 |
| `src/childsafe/web` | python | 38 | 25 |

## Extraction Quality

- Files parsed: **206** / 39301 seen (63 file(s) + 15 directory tree(s) skipped by ignore globs)
- Parse errors: 0

## Architecture Pattern

_No specific architecture pattern detected._

## Entry Points

- **main**: `scripts/android_toolchain.main` (scripts/android_toolchain.py)
- **main**: `scripts/audit_audio_labels.main` (scripts/audit_audio_labels.py)
- **main**: `scripts/audit_pen_baseline.main` (scripts/audit_pen_baseline.py)
- **main**: `scripts/b1_normalization.main` (scripts/b1_normalization.py)
- **main**: `scripts/benchmark_classifiers.main` (scripts/benchmark_classifiers.py)
- **main**: `scripts/benchmark_optimizations.main` (scripts/benchmark_optimizations.py)
- **main**: `scripts/benchmark_runtime.main` (scripts/benchmark_runtime.py)
- **main**: `scripts/build_detector_manifest.main` (scripts/build_detector_manifest.py)
- **main**: `scripts/check_demo_readiness.main` (scripts/check_demo_readiness.py)
- **main**: `scripts/collect_openimages.main` (scripts/collect_openimages.py)
- **main**: `scripts/collect_openimages_detector.main` (scripts/collect_openimages_detector.py)
- **main**: `scripts/companion_certificate.main` (scripts/companion_certificate.py)
- **main**: `scripts/companion_relay.main` (scripts/companion_relay.py)
- **main**: `scripts/companion_webcam.main` (scripts/companion_webcam.py)
- **main**: `scripts/compare_ood.main` (scripts/compare_ood.py)
- **main**: `scripts/diagnose_session.main` (scripts/diagnose_session.py)
- **main**: `scripts/download_childsun.main` (scripts/download_childsun.py)
- **main**: `scripts/download_mediapipe_models.main` (scripts/download_mediapipe_models.py)
- **main**: `scripts/evaluate_choking.main` (scripts/evaluate_choking.py)
- **main**: `scripts/export_android_models.main` (scripts/export_android_models.py)
- **main**: `scripts/extract_features.main` (scripts/extract_features.py)
- **main**: `scripts/fetch_openimages.main` (scripts/fetch_openimages.py)
- **main**: `scripts/finalize_session.main` (scripts/finalize_session.py)
- **main**: `scripts/inspect_dataset.main` (scripts/inspect_dataset.py)
- **main**: `scripts/install_android_sdk.main` (scripts/install_android_sdk.py)
- **main**: `scripts/label_tool.main` (scripts/label_tool.py)
- **main**: `scripts/make_splits.main` (scripts/make_splits.py)
- **main**: `scripts/make_vapid_keys.main` (scripts/make_vapid_keys.py)
- **main**: `scripts/package_android_models.main` (scripts/package_android_models.py)
- **main**: `scripts/plan_camera_trials.main` (scripts/plan_camera_trials.py)
- **main**: `scripts/play_area_demo.main` (scripts/play_area_demo.py)
- **main**: `scripts/prepare_expanded_data.main` (scripts/prepare_expanded_data.py)
- **main**: `scripts/prepare_fair_evidence.main` (scripts/prepare_fair_evidence.py)
- **main**: `scripts/prepare_full_training.main` (scripts/prepare_full_training.py)
- **main**: `scripts/profile_live_session.main` (scripts/profile_live_session.py)
- **main**: `scripts/run_webcam.main` (scripts/run_webcam.py)
- **main**: `scripts/set_kaggle_token.main` (scripts/set_kaggle_token.py)
- **main**: `scripts/train_all.main` (scripts/train_all.py)
- **main**: `scripts/web_demo.main` (scripts/web_demo.py)
- **main**: `src/childsafe/classifier/evaluate.main` (src/childsafe/classifier/evaluate.py)
- **main**: `src/childsafe/classifier/gradcam.main` (src/childsafe/classifier/gradcam.py)
- **main**: `src/childsafe/classifier/train.main` (src/childsafe/classifier/train.py)
- **main**: `src/childsafe/evaluation/cli.main` (src/childsafe/evaluation/cli.py)
48 routes, 15 shown:
- **route** DELETE `/api/capture` (src/childsafe/web/label_tool.py)
- **route** DELETE `/v1/rooms/{room}/cameras/{camera}` (src/childsafe/companion/relay.py)
- **route** DELETE `/v1/rooms/{room}` (src/childsafe/companion/relay.py)
- **route** GET `/` (services/notify/app.py)
- **route** GET `/` (src/childsafe/web/app.py)
- **route** GET `/` (src/childsafe/web/label_tool.py)
- **route** GET `/` (src/childsafe/web/play_area.py)
- **route** GET `/api/annotations.csv` (src/childsafe/web/app.py)
- **route** GET `/api/events` (services/notify/app.py)
- **route** GET `/api/health` (services/notify/app.py)
- **route** GET `/api/info` (src/childsafe/web/app.py)
- **route** GET `/api/info` (src/childsafe/web/label_tool.py)
- **route** GET `/api/pairing` (services/notify/app.py)
- **route** GET `/api/phone/qr.svg` (src/childsafe/web/app.py)
- **route** GET `/api/phone` (src/childsafe/web/app.py)
- … and 33 more (query_facts(kind="route") for all)

## Routes

| Method | Path | File | Type |
|--------|------|------|------|
| GET | `/` | `services/notify/app.py` |  |
| GET | `/` | `src/childsafe/web/app.py` |  |
| GET | `/` | `src/childsafe/web/label_tool.py` |  |
| GET | `/` | `src/childsafe/web/play_area.py` |  |
| POST | `/api/alert` | `services/notify/app.py` |  |
| POST | `/api/annotation` | `src/childsafe/web/app.py` |  |
| GET | `/api/annotations.csv` | `src/childsafe/web/app.py` |  |
| POST | `/api/camera/end` | `src/childsafe/web/app.py` |  |
| POST | `/api/camera/renew` | `src/childsafe/web/app.py` |  |
| POST | `/api/camera/start` | `src/childsafe/web/app.py` |  |
| POST | `/api/camera/sync` | `src/childsafe/web/app.py` |  |
| POST | `/api/capture` | `src/childsafe/web/label_tool.py` |  |
| DELETE | `/api/capture` | `src/childsafe/web/label_tool.py` |  |
| GET | `/api/events` | `services/notify/app.py` |  |
| POST | `/api/frame` | `src/childsafe/web/app.py` |  |
| GET | `/api/health` | `services/notify/app.py` |  |
| GET | `/api/info` | `src/childsafe/web/app.py` |  |
| GET | `/api/info` | `src/childsafe/web/label_tool.py` |  |
| GET | `/api/pairing` | `services/notify/app.py` |  |
| GET | `/api/phone` | `src/childsafe/web/app.py` |  |
| GET | `/api/phone/qr.svg` | `src/childsafe/web/app.py` |  |
| GET | `/api/play-area/health` | `src/childsafe/web/play_area.py` |  |
| GET | `/api/play-area/info` | `src/childsafe/web/play_area.py` |  |
| POST | `/api/play-area/scan` | `src/childsafe/web/play_area.py` |  |
| POST | `/api/reset` | `src/childsafe/web/app.py` |  |
| POST | `/api/restore` | `services/notify/app.py` |  |
| POST | `/api/review` | `src/childsafe/web/label_tool.py` |  |
| GET | `/api/review` | `src/childsafe/web/label_tool.py` |  |
| GET | `/api/session.csv` | `src/childsafe/web/app.py` |  |
| GET | `/api/session.json` | `src/childsafe/web/app.py` |  |
| POST | `/api/subscribe` | `services/notify/app.py` |  |
| GET | `/api/subscriptions` | `services/notify/app.py` |  |
| POST | `/api/test` | `services/notify/app.py` |  |
| GET | `/api/vapid` | `services/notify/app.py` |  |
| GET | `/files/{rel:path}` | `src/childsafe/web/label_tool.py` |  |
| GET | `/label-tool.js` | `src/childsafe/web/label_tool.py` |  |
| GET | `/play-area` | `src/childsafe/web/play_area.py` |  |
| GET | `/play-area.js` | `src/childsafe/web/play_area.py` |  |
| GET | `/sw.js` | `services/notify/app.py` |  |
| POST | `/v1/cameras/{camera}/events` | `src/childsafe/companion/relay.py` |  |
| POST | `/v1/cameras/{camera}/session` | `src/childsafe/companion/relay.py` |  |
| POST | `/v1/join` | `src/childsafe/companion/relay.py` |  |
| POST | `/v1/rooms` | `src/childsafe/companion/relay.py` |  |
| DELETE | `/v1/rooms/{room}` | `src/childsafe/companion/relay.py` |  |
| DELETE | `/v1/rooms/{room}/cameras/{camera}` | `src/childsafe/companion/relay.py` |  |
| GET | `/v1/rooms/{room}/events` | `src/childsafe/companion/relay.py` |  |
| POST | `/v1/rooms/{room}/invitations` | `src/childsafe/companion/relay.py` |  |
| GET | `/v1/rooms/{room}/snapshot` | `src/childsafe/companion/relay.py` |  |

## Dependency Rules

- `.` -> `src/childsafe/alerts`
- `.` -> `src/childsafe/companion`
- `.` -> `src/childsafe/data`
- `.` -> `src/childsafe/evaluation`
- `.` -> `src/childsafe/vision`
- `.` -> `src/childsafe/web`
- `.` -> `src/childsafe`
- `android/app/src/main/kotlin/vn/childsafe/app` -> `android/core/src/main/kotlin/vn/childsafe/core`
- `android/app/src/test/kotlin/vn/childsafe/app` -> `android/core/src/main/kotlin/vn/childsafe/core`
- `scripts` -> `src/childsafe/classifier`
- `scripts` -> `src/childsafe/runtime`
- `src/childsafe` -> `src/childsafe/alerts`
- `src/childsafe` -> `src/childsafe/classifier`
- `src/childsafe` -> `src/childsafe/data`
- `src/childsafe` -> `src/childsafe/evaluation`
- `src/childsafe` -> `src/childsafe/runtime`
- `src/childsafe` -> `src/childsafe/vision`
- `src/childsafe` -> `src/childsafe`
- `src` -> `src/childsafe/evaluation`
- `src` -> `src/childsafe/runtime`
- `src` -> `src/childsafe/vision`

## Critical Modules

| Module | Fan-In | Fan-Out | Criticality |
|--------|--------|---------|-------------|
| `src/childsafe` | 6 | 17 | high |
| `src/childsafe/runtime` | 8 | 0 | medium |
| `scripts` | 0 | 7 | medium |
| `src/childsafe/vision` | 6 | 0 | medium |
| `android/core/src/main/kotlin/vn/childsafe/core` | 5 | 0 | medium |
| `src/childsafe/data` | 4 | 0 | low |
| `src/childsafe/evaluation` | 4 | 0 | low |
| `android/app/src/main/kotlin/vn/childsafe/app` | 0 | 3 | low |
| `android/app/src/test/kotlin/vn/childsafe/app` | 0 | 2 | low |
| `src/childsafe/alerts` | 2 | 0 | low |

## Risk Zones

- **Cyclic dependency detected (7 modules)** (confidence: 100%): The following modules form a dependency cycle: src/childsafe -> src/childsafe/alerts -> src/childsafe/classifier -> src/childsafe/data -> src/childsafe/evaluation -> src/childsafe/runtime -> src/childsafe/vision -> src/childsafe. This can cause initialization issues, make refactoring harder, and indicates tight coupling.

---

*Generated at 2026-11-08T04:22:23Z in 7m37.6692247s. 2434 facts, 114 insights.*
