# Runtime optimization — 2026-10-12

Based on colleague commit `88f40ac` on `origin/main`. Keeps the existing detector, classifier,
multi-person tracker, camera UI and alert thresholds. No retraining, dataset expansion or threshold tuning.

## Changes

- A different detector label invalidates a held-object match even if boxes overlap. Object changes,
  ambiguous ownership, classification errors and unclassifiably small crops clear cached predictions.
  A previous unsafe classification cannot be used as evidence for a new/unclassified object.
- Classifier errors belong to each person. A bad crop for one person no longer marks other people’s
  classifier unavailable; the aggregate diagnostic still exposes any failure.
- Skip the zoom detector pass when the full-frame detection already gives an unambiguous held mask,
  enough mask confidence/area, and mask proximity below the existing exit distance. Missing, weak,
  unassigned or distant masks still use zoom. `detector.roi_skip_resolved: false` restores the previous
  always-zoom policy for controlled comparisons. This optimization can change which detection wins;
  it is not a claim of equivalent accuracy or complete object coverage.
- Bound the landmarker pool to `runtime.max_people` rather than allowing one extra pooled instance.
- Execute web inference in a worker thread so it cannot block the ASGI event loop. Concurrent frame
  uploads receive HTTP 503 with a retry header instead of waiting behind old frames. The existing
  browser loop sends a fresh frame on its next iteration and clears old error text after recovery.
- Optional `runtime.cpu_threads` is respected even after Ultralytics lazily initializes its CPU
  backend. Default **0** retains library defaults; choose a nonzero budget only after benchmarking
  the target machine. Training does not call the CLI runtime configuration helper.
- Regenerate both classifier notebooks to keep their embedded config synchronized. Ignore the local
  supplemental images so they cannot be accidentally staged.

## Reproducible performance measurement

Run from the repository root:

```powershell
.venv\Scripts\python scripts\benchmark_optimizations.py --ckpt outputs/with_other/runs/efficientnet_b0_group_robust/best.pt
```

The benchmark performs real YOLO and classifier inference, but injects synthetic person, mouth,
hand and object-mask fixtures. It excludes actual MediaPipe landmark inference and does not use
images of people. It measures the **resolved-object workload**, not webcam accuracy or camera latency.
Each variant has three warm-up frames and 15 measured frames. Output is
`outputs/optimization_benchmark.json` (ignored by Git), including model hashes and scope limitations.

Measured in the project’s CPU `.venv` with Torch `2.6.0+cpu`, eight threads selected by the detector:

| Policy | Detector calls/frame | Median | p90 |
|---|---:|---:|---:|
| Previous: always zoom | 2 | 182.14 ms | 185.85 ms |
| Optimized: skip resolved mask | 1 | 96.05 ms | 102.44 ms |

Median decreased by approximately **47%** in this workload. Unresolved objects still require the
extra detector pass. These timings do not demonstrate higher event recall or fewer false alarms.
Results vary with CPU budget and load; a two-thread experiment was slower on this machine, so the
default does not force that budget.

## Validation and local setup

- Regression tests cover replacement of a knife with a spoon at the same location, weak-mask zoom,
  classifier failure/recovery, independent per-person health, small-crop rejection, responsive info
  requests during inference, HTTP 503 for overlapping frames and CPU backend thread-budget reset.
- The full suite: **117 passed, 3 skipped**. The opt-in real-model check: **1 passed**, with a reviewed, object-only knife
  crop, local YOLO segmentation, classifier and MediaPipe models. Default integration tests that need
  explicitly supplied images/checkpoints remain skipped rather than being silently simulated.
- The checkout has its own `.venv`, installed from `requirements-runtime.txt`, and an editable
  ChildSafe installation. This is CPU Torch; GPU performance is not claimed. Dependencies and model
  files remain untracked. The detector was copied from the colleague’s local worktree and verified
  against `docs/model_provenance.json` before use.
- The actual launcher was tested on temporary port 8765: `/api/info` returned HTTP 200 and a real
  object-image request reported both detector/classifier healthy. The test service was stopped after
  verification; no webcam was opened. The fallback live model is the existing robust + other
  checkpoint because full-data Kaggle weights are not present in this checkout.

Start the local web demo with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_web_demo.ps1
```

Open `http://127.0.0.1:8000`, then choose Start camera. No camera is opened by the tests or benchmark.
Stop the service using `scripts/stop_web_demo.ps1`.

Not yet verified: live elevated-view adult/doll scenarios, multi-person event recall, false alarms/hour
or accuracy with pens/coins. The existing uncalibrated/unsupported-object limitations remain visible.
