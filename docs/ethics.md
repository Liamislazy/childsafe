# Ethics and data handling (from handoff section 8)

- **No video of real children.** Adults and dolls act out scenarios with plastic / mock, non-sharp, non-swallowable objects.
- Processing is on-device. `run_webcam.py` never writes video. It writes an event CSV (time, level, confidence, object class)
  and, only with `--log-features`, per-frame numbers (distances, flags). No images, no raw landmarks.
- Evaluation recordings (`data/recordings/`, git-ignored) only from adult volunteers with written consent covering purpose,
  retention period, and the right to withdraw. Delete recordings after the project; feature CSVs contain no images.
- No cloud VLM / API on frames that contain faces.
- The system does **not** replace adult supervision. Worst failure mode: a missed danger (false negative), e.g. when the
  object is occluded, the lighting is poor, or the child is outside the camera view. Say this in the report and in the demo.
- Landmark models were trained mostly on adults; results with adults and dolls do not transfer automatically to young children.
- Legal (partly unverified, check original texts before any data collection involving children): from 01/01/2026 the
  Personal Data Protection Law 91/2025/QH15 and Decree 356/2025/NĐ-CP apply in Vietnam (Decree 13/2023 no longer applies).
- Ask the school whether ethics approval is required for adult volunteer video before recording.
