# Scenario annotations (E2)

Only adult volunteers (written consent) and dolls, with plastic/mock objects. See docs/ethics.md.
Videos go in `data/recordings/` (git-ignored) and are deleted at the end of the project.

`annotations.csv` — one row per annotated segment, times in seconds from the start of the video:

```
video,start,end,label
p01_light_cam80_yaw0,12.4,15.9,unsafe_mouthing
p01_light_cam80_yaw0,40.0,52.3,eating
```

Labels
- `unsafe_mouthing` — an unsafe mock object (plastic knife, fork, scissors...) reaches the mouth. **Positive event.**
- `safe_mouthing`, `eating`, `drinking`, `nose_scratch`, `chin_rest`, `face_touch` — non-dangerous; used to break down false alarms.

`video` must equal the video file name without extension (the feature CSV name).

`videos.csv` — which videos are for tuning (`dev`) and which for the final report (`test`).
Split by **person**, not by clip, so the same volunteer never appears in both.

```
video,split
p01_light_cam80_yaw0,dev
p05_dim_cam120_yaw30,test
```

Export from ELAN: File > Export As > Tab-delimited text (begin/end time in seconds, annotation value), then rename columns.
Export from CVAT: use a video "tag" track per event and convert frame numbers to seconds (frame / fps).
