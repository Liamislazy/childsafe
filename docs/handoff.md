# Handoff: Child dangerous-object warning system (computer vision)

> Put this file in the project repo (e.g. `docs/handoff.md`) or copy the key parts into `CLAUDE.md`.
> Written from a long claude.ai chat. Facts are tagged by how well they were verified. Do not treat [UNVERIFIED] items as facts.

## 0. Instructions for Claude Code

- The user is a student in Ho Chi Minh City, Vietnam. **Reply in Vietnamese by default**, switch to English if the user writes in English.
- Do not invent numbers, DOIs, authors or links. If you cannot verify something, say so.
- Keep reported results honest. A lower number after fixing a flaw is a valid finding, not a failure.
- Prefer the scoped-down plan in section 3. Do not add heavy components (VLM, depth cameras, SAHI) unless the user asks.
- Never collect video of real children. See section 7.

## 1. Project in one paragraph

A real-time warning system for dangerous objects near young children. **Module A** classifies objects as safe/unsafe using the ChildSUn dataset. **Module B** watches a single webcam, finds the hand/held object and the mouth (MediaPipe), and computes a normalized distance and approach speed. A **rule layer** combines the object label, distance, speed and time into three alert levels: normal / caution / danger.

Working title: *Hệ thống cảnh báo đồ vật nguy hiểm cho trẻ nhỏ dựa trên thị giác máy tính: kết hợp phân loại đồ vật an toàn/không an toàn và giám sát khoảng cách tới miệng theo thời gian thực.*

Constraints: one semester (about 12-14 weeks), Colab and a laptop with a webcam, no expensive hardware, no data from real children.

## 2. History and decisions

- The original idea was toy safety classification. It was narrowed because most toy hazards (lead, phthalates) are not visible in photos.
- **Important correction:** ChildSUn is **not a toy dataset**. It is a household-objects dataset. Toys are only one of 10 categories, on the safe side.
- The user chose to keep ChildSUn and reframe the topic as **dangerous objects for children** (option 1 of three).
- Other topics were explored and dropped because the user found them infeasible: Vietnamese sign language, scam-message detection, flood depth, dengue breeding sites, child passengers on motorbikes, product label compliance, medicine-label OCR, home sound alerts, PM2.5 forecasting (not novel enough). The user then chose to **return to the children topic**.

## 3. Agreed scope (scoped-down version)

- Adults or dolls act out scenarios. Use plastic or mock objects (plastic knife). No real children.
- **Module A:** train on ChildSUn only. Compare 2-3 lightweight models (EfficientNet-B0, MobileNetV2/V3, VGG19 as baseline). Add Grad-CAM.
- **Module B:** one webcam plus MediaPipe Face Mesh/Hands. Distance normalized by face width, which is a **relative** distance. State this as a limitation.
- **Rule layer:** simple rules, no third trained model.
- **Main contribution:** evaluation. Report false alarms per hour versus a distance-only baseline, plus the performance drop on out-of-domain images.

## 4. ChildSUn dataset

- Kaggle: `kaggle.com/datasets/fairuzkhan002/datasetvgg19` (the Kaggle page only returned metadata when fetched; class counts, license and file structure are **not yet checked**).
- Paper: Khan & Dey (2024), "Towards Enhancing Child Safety: A Deep Learning Approach to Detect Child Safe and Unsafe Objects", IEEE WIECON-ECE 2024, pp. 123-128, DOI 10.1109/WIECON-ECE64149.2024.10914992. Authors and DOI checked via Crossref. **Only the abstract was read.**
- Reported: about 1,358 raw images grown to about 5,350; 10 categories (knife, socket, scissors, electric heater, fork, toy, ball, baby items, fruit, spoon) grouped into safe/unsafe; custom CNN plus six pretrained models compared; EfficientNet-B0 reported at 99.63% on the test set.
- Suspicion [inference, unverified]: augmentation may have been applied before the train/test split, causing leakage. The paper's split procedure was not read. Check this against the full text before claiming it.
- Other gaps [inference]: image classification only, no out-of-domain test, no Grad-CAM, no small swallowable objects (button batteries, coins, magnets).

## 5. Literature: what was verified

Reading level: **FULL** = full text read, **ABS** = abstract/snippet only.

| Work | Level | Key facts |
|---|---|---|
| Ahmad et al. 2025, IJIST 7(SI ICTIS-2025):297-306 | FULL | YOLOv8-large, 8 classes (baby, pool, stairs, knife, broken glass, socket, medicine, reptile). mAP 90%. Alert when the box-center distance exceeds a fixed 700 px threshold; SMS via SIM800L. Image counts inconsistent in the paper (over 30,000; over 32,000 train + over 5,000 val; table sums to 39,000). No independent test set found. Knife class 84%. Authors admit occasional false positives and lighting sensitivity. |
| Ramadan et al. 2025, Indonesian J. Innovation Studies 26(4), DOI 10.21070/ijins.v26i4.1851 | ABS | YOLOv11n detects hand/face/object, hand-to-mouth Euclidean distance. Accuracy 92% without an object, 74% with an object. Dataset, FPS, threshold units [UNKNOWN]. Four authors: Ramadan, Muhtadi, Rafi, Waluyo. Read the full PDF. |
| Lupolt et al. 2025, J Expo Sci Environ Epidemiol 36(4):725-733, DOI 10.1038/s41370-025-00814-x | FULL | 4 GoPro Hero9 cameras, children 6-18 months. Pipeline: Faster R-CNN, HRNet (133 keypoints), Grounded SAM, EasyMocap 3D, distance threshold D = 6 cm, ResNet50 for hand-vs-object mouthing. Temporal rule: more than 3 frames per second, merge events within 4 s. Per-second agreement 96.9-99.9%. Count error for object-to-mouth rises from 2.18 (4 cameras) to 4.72 (3) and 5.10 (2). Results barely change for D between 4 and 10 cm. Offline, up to about 5 h preprocessing plus 20 h code per set. Mean 27 object-to-mouth and 3 hand-to-mouth events per hour. Sample counts are inconsistent inside the paper. |
| Lupolt et al. companion, DOI 10.1038/s41370-025-00800-3 | ABS | 61 children, 20-minute videos in two play scenarios. A "median 23 object-to-mouth per hour" figure came up in the chat but was **not confirmed** [UNVERIFIED]. |
| Finocchiaro et al. 2025, arXiv 2507.13326 | FULL | Cascade: contact-action module (Mamba with a distilled EfficientNetV2) triggers object detection (YOLO-World). p-AP 38.52% at 30 fps (30 fps is the **evaluation** frequency, not throughput). Contact module 0.007 s/frame; full pipeline about 0.064 s/frame (about 15 fps, my calculation). Best IoU threshold 0.01. Industrial egocentric data (ENIGMA-51). |
| KidRisk, arXiv 2606.25298 (Nguyen, Do, Phung, Dao, Tran, Le) | ABS | 2,500 videos and 10,000 images of dangerous child actions. VLM baseline 83.53% (action) and 96.14% (dangerous action). Action level, not object level. |
| Anemeta & Rzepka 2025, Algorithms 18(8):507, DOI 10.3390/a18080507 | ABS | Compares VLMs and humans on hazard judgment in child environments. First author is Anemeta. |
| AAP policy 2010, Pediatrics 125(3):601 | ABS | Choking is a leading cause of injury and death in young children. |

Foundational tools (not re-read, cite only as method descriptions): EfficientNet (Tan & Le, ICML 2019), MobileNetV2/V3, Grad-CAM (Selvaraju et al., ICCV 2017), MediaPipe Face Mesh (arXiv 1907.06724), MediaPipe Hands (arXiv 2006.10214).

**The user's own reference list has errors.** Do not cite from it without checking. Examples found: KidRisk's real title differs from the one listed; several links point to unrelated papers (a sound-event review, an autonomous-vehicle index, a radar dataset, a child-robot autism paper, an urban-street study, a BLE tracking paper); the HEROCS link duplicated the Khan & Dey link; the SSPS PLOS ONE link returned 404. HEROCS (ICICT 2026) exists but its content was not read.

## 6. Improvement plan and difficulty

### Minimum viable package (about 12-14 weeks, all Easy/Medium)

Priority order: D1, E1, A1, B1, D2, A3, A2, B2, D3, A5, E2, E4.

| ID | Item | Difficulty | Weeks |
|---|---|---|---|
| A1 | Leak-free split: perceptual hash (pHash/dHash), group near-duplicates, `GroupShuffleSplit`/`StratifiedGroupKFold`; report random split and group split side by side | Easy | 1 |
| A2 | Out-of-domain test set: 300-600 webcam photos of the same 10 classes, 3 light levels, with and without a hand, 2-3 distances | Easy-Medium | 2 |
| A3 | Temperature scaling on the val set, report ECE and reliability diagrams in and out of domain, reject threshold (low confidence becomes "caution") | Easy | 1 |
| A5 | Augmentation against occlusion, blur, low light (random erasing, motion blur, gamma, ISO noise) plus a **noise-severity scale** and degradation curve | Easy | 1-2 |
| B1 | Compare distance normalization: face width vs inter-eye distance vs hand size; measure the coefficient of variation across camera distances and head rotation | Easy | 1 |
| B2 | Smoothing (EMA / One Euro / Kalman); if the mouth landmark is lost while the hand was just near the face, hold the last state for 0.5-1 s and raise "caution". **Treat lost mouth landmarks as a signal.** | Medium | 1-2 |
| D1 | Temporal rule: N-of-M frames (Lupolt used more than 3 in 10), hysteresis (enter threshold smaller than exit), merge events within T seconds, cooldown after a danger alert | Easy | 1 |
| D2 | Cascade: run Module A only when normalized distance is below a "far" threshold | Easy-Medium | 1 |
| D3 | Alert matrix from object label (safe/unsafe/uncertain), calibrated confidence, distance, approach time, dwell time, landmark-loss flag. Safe objects cap at "caution". Tune thresholds by grid search on a dev set, keep a separate test set | Easy | 1 |
| E1 | Metrics: false alarms per hour on non-dangerous segments (eating, drinking, scratching nose, chin on hand), event recall with a 1-3 s tolerance, alert latency (median, p90), event count error, per-second agreement, trade-off curve vs a per-frame distance-only baseline | Easy-Medium | 1-2 |
| E2 | Scripted scenarios: 5-8 adult volunteers and 1-2 dolls, at least 1-2 hours of video, 3 light levels, 2-3 distances, head rotation 0/30/60 degrees; annotate event start/end (ELAN or CVAT) | Medium | 2 |
| E4 | Ablation: switch each component off in turn | Easy | 1 |

Suggested schedule: weeks 1-2 A1 and retrain 2-3 lightweight models; week 3 A3; weeks 3-4 A5; weeks 4-5 A2; weeks 5-6 B1; weeks 6-7 B2; weeks 7-8 D1, D2, D3; weeks 8-10 E2; weeks 10-12 E1 and E4; weeks 12-14 Grad-CAM (qualitative) and write-up.

### Extension package (if 4-8 weeks remain)

A4 (MC Dropout or a 3-model ensemble), A7 (classify a crop around the hand from MediaPipe Hands), B3 (approach velocity and time-to-contact), A6 (quantitative Grad-CAM with boxes on 100-200 images), A10 (INT8/ONNX quantization), E3 (evaluation by doll size / adult-trained landmark gap).

### Future work (do not attempt now)

A8 (YOLO with SAHI, needs box labels), A9 (small swallowable object class, needs new data and safe mock objects), B4 (monocular depth, only relative), B6 (learned contact classifier), D4 (VLM re-check, privacy and latency problems), B5 (depth camera, hardware).

## 7. Technical risks

1. **Monocular depth ambiguity.** A 2D normalized distance can be small when the hand is in front of the face but far from the mouth. Mitigation: camera placement, tie to hand/face size ratio and MediaPipe relative z, temporal rule, state it as a limitation.
2. **Occlusion.** Ramadan's accuracy fell from 92% to 74% with an object in hand. Mitigation: B2, A5, A7, the "uncertain" class in D3.
3. **Domain gap** between ChildSUn images and webcam frames. Mitigation: A2, A3, A5, A6. If the drop is large, say plainly that ChildSUn alone is not enough.
4. **False alarms and alarm fatigue.** Children mouth objects often (about 27 per hour in Lupolt), mostly safe objects. Mitigation: D1, D3, and make false alarms per hour the headline metric.
5. **Landmark models were mostly trained on adults.** Lupolt found HRNet worked on 6-18 month olds without retraining, but only in a 4-camera setup. Keep conclusions limited to adults and dolls and say so.

## 8. Ethics and legal

- Process on the device. **Do not store video** by default. Log events only (time, alert level, confidence).
- Evaluation video only from adult volunteers with written consent (purpose, retention, right to withdraw). Delete after the project.
- Use dolls and non-sharp, non-swallowable mock objects. No cloud VLM or API on frames containing faces.
- Say clearly that the system does not replace adult supervision. Describe its worst failure mode (missed danger).
- Legal, [PARTLY UNVERIFIED]: Decree 13/2023 no longer applies. From 01/01/2026 the Personal Data Protection Law 91/2025/QH15 and Decree 356/2025/NĐ-CP apply. Details for children (consent of the legal representative, and of the child from 7 years old for some uses) came from secondary summaries only. **Check the original texts** before collecting any data from children.

## 9. Novelty claims and their limits

Four points that were not found in the searches: (1) false alarms per hour and alert latency for a single-webcam hand/object-to-mouth warning system versus a distance-only baseline; (2) a leak-controlled re-evaluation of ChildSUn with an out-of-domain webcam test and calibration; (3) combining calibrated object class, normalized distance and a temporal rule into three alert levels; (4) using a lost mouth landmark near the hand as a caution signal.

"Not found" is not "nobody did it". The search covered about 18 web queries plus the papers listed above. IEEE Xplore, Scopus and Google Scholar were not searched systematically. Search them before claiming novelty.

## 10. Files from the chat

- `Tong_quan_tai_lieu.docx`: Vietnamese literature review chapter (Word). It contains the corrected "how each paper solves the problem" section.
- Three research reports were produced as chat artifacts (download them from the chat): the detailed literature review with appendices of excluded and unverified sources; the evaluation of 8 alternative topics; the improvement options with difficulty ratings.
- Not carried over: the chat transcript itself. This file is the summary.

## 11. First tasks for Claude Code

1. Set up the repo: `data/`, `notebooks/`, `src/`, `docs/`, `requirements.txt`.
2. Download ChildSUn (Kaggle API or manual). Record the license, class counts, folder structure, and whether augmented images are already included.
3. Compute perceptual hashes, find near-duplicate groups, and build a **group-based** train/val/test split. Save the split indices.
4. Train EfficientNet-B0, MobileNetV3 and VGG19 (baseline) on that split. Report accuracy, precision, recall and F1 per class, and recall of the unsafe class.
5. Compare with a plain random split to quantify possible leakage.
6. Add temperature scaling and reliability diagrams.
7. Only then start Module B (MediaPipe landmarks, normalized distance, smoothing) and the temporal rule.

Open questions to ask the user before starting: Colab or local GPU? Is a webcam available? Does the school require ethics approval for adult volunteer video?
