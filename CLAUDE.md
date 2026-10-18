# CLAUDE.md

Full context: docs/handoff.md (read it first). Key rules:

- User is a student in Ho Chi Minh City. Reply in Vietnamese by default; English if the user writes in English.
- Never invent numbers, DOIs, authors or links. Tag unverified items. Keep results honest: a lower score after fixing a flaw (e.g. leakage) is a finding.
- Scoped-down plan only (handoff section 3/6). No VLM, depth camera, SAHI unless asked.
- Never collect or process video of real children (docs/ethics.md). No video written to disk by the app.
- Never tune thresholds on the test split; use `evaluate_alerts.py grid --sessions outputs/scenarios --split dev` and the shared explicit-exposure metrics. `--legacy-metrics` is for historical reports only.

Environment: Windows, Python 3.12 venv in `.venv`. Last verified 18/10/2026: torch 2.6.0+cpu; GPU runtime has not been validated in this environment.
Run tests: `.venv\Scripts\python -m pytest -q`. MediaPipe >= 0.10.2x has no `mp.solutions`; we use the Tasks API with
`.task` files in `models/` (scripts/download_mediapipe_models.py). Pooled per-person tasks use stateless IMAGE mode so hidden VIDEO tracker history cannot transfer between people. See docs/product_runtime_guide.md and docs/review_changes_2026_10_18.md for the current scope and unverified real-camera metrics.

ChildSUn facts (checked 2026-10-01, data in data/raw/childsun):
- Flat folder, 5,353 images 224x224, class = filename prefix (typos: `seissor`, `socket` + `sockets` sets). License on Kaggle: "unknown".
- Augmentation is pre-applied: `<class>_<srcIdx>_0_<rand>.jpg` = 5 copies per source photo; 1,358 source photos
  (matches the paper's ~1,358 raw). Fruits have no copies but contain exact duplicate files under different names.
- Kaggle description puts forks under unsafe.
- Random split: 91.8% of test images share a source photo with train. Group split (filename source + pHash<=8 & dHash<=5): 0%.
- pHash alone did not link the augmented copies (strong geometric aug), so filename grouping is essential.
- Canonical group split checksum: `42e3fb2547c5` (own greedy splitter; sklearn StratifiedGroupKFold differed across versions).
- Runs before 2026-10-01 afternoon (outputs/runs, outputs/kaggle) used older, different splits and VGG lr 3e-4
  (diverged, best epoch 0): do not report them; retrain all 6 on the canonical split.

Open items: ethics approval question for the school; full text of Khan & Dey to see how they split.
