"""Fine-tune the segmentation detector on frames from the demo webcam (adults/dolls only).

Subcommands, in order:
  capture   save still frames (never video) from the webcam into one session folder
  prelabel  write teacher pseudo-labels (default yolo11m-seg) in YOLO-seg format
  review    accept / flag each pseudo-label by eye; accepted ones become reviewed labels
  build     assemble a dataset; whole sessions go to the split chosen at capture time
  train     fine-tune yolo11n-seg and compare it with the base model on the held-out test split

Frames stay in data/recordings/ (git-ignored). Keep all 80 COCO class ids so the result is a
drop-in replacement for models/yolo11n-seg.pt (the runtime matches labels by name).
"""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('data/recordings/detector_frames')
SPLITS = ('train', 'val', 'test')
REVIEW_KEYS = {'a': 'accepted', 'f': 'needs_fix', 'x': 'excluded'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygon_line(class_id, polygon):
    """One YOLO-seg row; polygon is normalized [[x, y], ...] with at least three points."""
    points = [(float(x), float(y)) for x, y in polygon]
    if len(points) < 3:
        raise ValueError('polygon needs at least three points')
    if any(not (0 <= v <= 1) for p in points for v in p):
        raise ValueError('polygon outside image')
    return f'{int(class_id)} ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in points)


def load_names(path):
    names = __import__('yaml').safe_load(Path(path).read_text(encoding='utf-8'))['names']
    return {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}


def frame_stats(bgr):
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return dict(brightness=round(float(gray.mean()), 2),
                sharpness=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2))


def capture(a):
    import cv2
    if not a.consent_confirmed:
        raise SystemExit('Only adult volunteers or dolls, with written consent. Pass --consent-confirmed.')
    session = a.root / a.session
    if session.exists():
        raise FileExistsError(f'{session} exists; choose a new --session')
    (session / 'images').mkdir(parents=True)
    meta = dict(session=a.session, split=a.split, participant=a.participant, conditions=a.condition,
                objects=a.objects, camera_index=a.camera, requested_size=[a.width, a.height],
                interval_s=a.interval, consent_confirmed=True,
                started_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit('camera did not open')
    rows, recording, last = [], False, 0.0
    print('SPACE start/pause saving, Q quit. Stop objects before the mouth.', flush=True)
    try:
        while len(rows) < a.max_frames:
            ok, frame = cap.read()
            if not ok:
                print('camera read failed', flush=True); break
            now = time.monotonic()
            if recording and now - last >= a.interval:
                name = f'{a.session}_{len(rows):05d}.jpg'
                # Quality 95: keep the camera's own artefacts without adding much new compression.
                cv2.imwrite(str(session / 'images' / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(dict(file=name, t=round(now, 3), width=frame.shape[1], height=frame.shape[0],
                                 **frame_stats(frame)))
                last = now
            view = frame.copy()
            cv2.putText(view, f"{'REC' if recording else 'PAUSED'} {len(rows)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (200, 200, 200), 2)
            cv2.imshow('capture (frames only)', view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): recording = not recording
            elif key == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows()
    meta.update(frames=len(rows), ended_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (session / 'session.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (session / 'frames.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ('session', 'split', 'frames')}), flush=True)


def prelabel(a):
    from ultralytics import YOLO
    model = YOLO(str(a.teacher), task='segment')
    for session in a.sessions:
        out = session / 'labels_prelabel'
        if out.exists():
            raise FileExistsError(f'{out} exists; pseudo-labels are written once per teacher')
        out.mkdir()
        images = sorted((session / 'images').glob('*.jpg'))
        for image in images:
            result = model.predict(str(image), conf=a.conf, imgsz=a.imgsz, retina_masks=True,
                                   device=a.device, verbose=False)[0]
            lines = []
            if result.masks is not None:
                for cls, poly in zip(result.boxes.cls.tolist(), result.masks.xyn):
                    if len(poly) >= 3:
                        lines.append(polygon_line(cls, poly.clip(0, 1)))
            # An empty file is a real negative frame, not a missing label.
            (out / f'{image.stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        info = dict(teacher=str(a.teacher), teacher_sha256=sha256(a.teacher), conf=a.conf, imgsz=a.imgsz,
                    images=len(images), status='unreviewed_pseudo_labels')
        (session / 'prelabel.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print(session.name, info['images'], 'frames pseudo-labelled', flush=True)


def draw_labels(bgr, label_path, names):
    import cv2
    import numpy as np
    h, w = bgr.shape[:2]
    for line in label_path.read_text(encoding='utf-8').splitlines():
        values = line.split()
        pts = (np.array(values[1:], dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.polylines(bgr, [pts], True, (0, 255, 0), 2)
        cv2.putText(bgr, names[int(values[0])], tuple(int(v) for v in pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
    return bgr


def review(a):
    """a = labels correct, f = needs manual fix (CVAT etc.), x = exclude frame with no fix, q = stop."""
    import cv2
    names = load_names(a.names) if a.names else None
    if names is None:
        from ultralytics import YOLO
        names = YOLO(str(a.teacher), task='segment').names
    session = a.session
    path = session / 'review.json'
    decisions = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    reviewed = session / 'labels_reviewed'; reviewed.mkdir(exist_ok=True)
    for image in sorted((session / 'images').glob('*.jpg')):
        if image.stem in decisions:
            continue
        label = session / 'labels_prelabel' / f'{image.stem}.txt'
        cv2.imshow('review: a=ok f=fix x=exclude q=quit', draw_labels(cv2.imread(str(image)), label, names))
        key = chr(cv2.waitKey(0) & 0xFF)
        if key == 'q':
            break
        if key not in REVIEW_KEYS:
            continue
        decisions[image.stem] = REVIEW_KEYS[key]
        if key == 'a':
            shutil.copy2(label, reviewed / label.name)
        path.write_text(json.dumps(decisions, indent=1), encoding='utf-8')
    cv2.destroyAllWindows()
    counts = {v: list(decisions.values()).count(v) for v in REVIEW_KEYS.values()}
    print(json.dumps(dict(session=session.name, **counts)), flush=True)


def label_source(session, stem, allow_unreviewed):
    """Return (label path, status) or (None, reason) for one frame."""
    reviewed = session / 'labels_reviewed' / f'{stem}.txt'
    if reviewed.exists():
        return reviewed, 'reviewed'
    decision = json.loads((session / 'review.json').read_text(encoding='utf-8')).get(stem) \
        if (session / 'review.json').exists() else None
    if decision == 'excluded':
        return None, 'excluded_by_reviewer'
    if decision == 'needs_fix':  # known-wrong pseudo-label: never train on it
        return None, 'needs_fix'
    pseudo = session / 'labels_prelabel' / f'{stem}.txt'
    if allow_unreviewed and pseudo.exists():
        return pseudo, 'pseudo_label'
    return None, 'unreviewed'


def build(a):
    import yaml
    out = a.out
    if (out / 'dataset.yaml').exists():
        raise FileExistsError('completed dataset exists; use --out with a new directory')
    names = load_names(a.names) if a.names else None
    if names is None:
        from ultralytics import YOLO
        names = YOLO(str(a.base), task='segment').names
    records, splits = [], {}
    for session in a.sessions:
        meta = json.loads((session / 'session.json').read_text(encoding='utf-8'))
        split = meta['split']
        if split not in SPLITS or not meta.get('consent_confirmed'):
            raise ValueError(f'{session.name}: invalid split or missing consent')
        if splits.setdefault(meta['session'], split) != split:
            raise ValueError(f'{session.name}: session listed twice with different splits')
        # Pseudo-labels may train, never validate or test: those numbers must come from human labels.
        allow = a.allow_unreviewed_train and split == 'train'
        for image in sorted((session / 'images').glob('*.jpg')):
            label, status = label_source(session, image.stem, allow)
            record = dict(session=meta['session'], split=split, participant=meta.get('participant', ''),
                          conditions='|'.join(meta.get('conditions') or []), file=image.name, label_status=status)
            if label is not None:
                for i, line in enumerate(label.read_text(encoding='utf-8').splitlines()):
                    values = line.split()
                    if int(values[0]) not in names or len(values) < 7 or len(values) % 2 == 0:
                        raise ValueError(f'{label}: bad row {i + 1}')
                dest = out / 'images' / split / image.name
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(image, dest)
                (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
                shutil.copy2(label, out / 'labels' / split / f'{image.stem}.txt')
                record['sha256'] = sha256(dest)
            records.append(record)
    used = [r for r in records if 'sha256' in r]
    seen = {}
    for r in used:
        if seen.setdefault(r['sha256'], r['split']) != r['split']:
            raise ValueError('identical frame content crosses splits')
    people = {s: {r['participant'] for r in used if r['split'] == s} for s in SPLITS}
    missing = [s for s in SPLITS if not any(r['split'] == s for r in used)]
    if missing:
        raise ValueError(f'empty split(s): {missing}; record and review at least one session for each')
    config = dict(path=str(out.resolve()), train='images/train', val='images/val', test='images/test',
                  names=dict(names))
    (out / 'dataset.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    with (out / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(records, f, indent=1)
    status = {s: {k: sum(r['split'] == s and r['label_status'] == k for r in records)
                  for k in sorted({r['label_status'] for r in records})} for s in SPLITS}
    report = dict(frames_used={s: sum(r['split'] == s for r in used) for s in SPLITS},
                  label_status=status, sessions={s: sorted({r['session'] for r in used if r['split'] == s})
                                                 for s in SPLITS},
                  participants_shared_between_train_and_test=sorted(people['train'] & people['test'] - {''}),
                  split_policy='whole sessions, split chosen at capture time',
                  annotation_type='YOLO-seg polygons, 80 COCO class ids')
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def metrics(result, names):
    per_class = {names[int(c)]: dict(precision=float(result.seg.p[i]), recall=float(result.seg.r[i]),
                                     mask_map50=float(result.seg.ap50[i]))
                 for i, c in enumerate(result.seg.ap_class_index)}
    return dict(summary={k: float(v) for k, v in result.results_dict.items()}, per_class=per_class)


def train(a):
    if a.output.exists():
        raise FileExistsError('training output exists; choose a new --output directory')
    from ultralytics import YOLO
    model = YOLO(str(a.base), task='segment')
    # Small own-camera sets overfit fast: low lr, frozen early backbone, early stopping on val.
    model.train(data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                lr0=a.lr0, freeze=a.freeze, workers=0, project=str(a.output.parent), name=a.output.name,
                exist_ok=False, seed=42, deterministic=True, patience=10, amp=False, plots=False,
                cache=False, save=True)
    best = Path(model.trainer.best)
    report = dict(base=str(a.base), base_sha256=sha256(a.base), checkpoint=str(best.resolve()),
                  checkpoint_sha256=sha256(best), epochs_requested=a.epochs, imgsz=a.imgsz, lr0=a.lr0,
                  freeze=a.freeze, data=str(a.data),
                  limitation='Own-camera adult/doll frames only; does not show transfer to children. '
                             'Other COCO classes are not re-measured (possible forgetting).')
    # Same held-out test sessions, same settings, for the base model and the fine-tuned one.
    for key, weights in (('base_test', a.base), ('finetuned_test', best)):
        evaluated = YOLO(str(weights), task='segment')
        result = evaluated.val(data=str(a.data), split='test', imgsz=a.imgsz, device=a.device, plots=False,
                               project=str(a.output), name=key)
        report[key] = metrics(result, evaluated.names)
    (a.output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k]['summary'] for k in ('base_test', 'finetuned_test')}, indent=2), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--session', required=True); c.add_argument('--split', choices=SPLITS, required=True)
    c.add_argument('--participant', required=True, help='code such as P01 or DOLL1, never a name')
    c.add_argument('--condition', nargs='*', default=[], help='e.g. dim backlight fast_motion')
    c.add_argument('--objects', nargs='*', default=[], help='props in this session, e.g. plastic_fork teddy')
    c.add_argument('--consent-confirmed', action='store_true')
    c.add_argument('--camera', type=int, default=0)
    c.add_argument('--width', type=int, default=1280); c.add_argument('--height', type=int, default=720)
    c.add_argument('--interval', type=float, default=0.5, help='seconds between saved frames')
    c.add_argument('--max-frames', type=int, default=600)
    c.add_argument('--root', type=Path, default=ROOT)
    c = sub.add_parser('prelabel')
    c.add_argument('sessions', type=Path, nargs='+')
    c.add_argument('--teacher', type=Path, default=Path('models/yolo11m-seg.pt'))
    c.add_argument('--conf', type=float, default=0.25); c.add_argument('--imgsz', type=int, default=960)
    c.add_argument('--device', default='cpu')
    c = sub.add_parser('review')
    c.add_argument('session', type=Path)
    c.add_argument('--teacher', type=Path, default=Path('models/yolo11m-seg.pt'))
    c.add_argument('--names', type=Path, help='dataset.yaml with names (skips loading the model)')
    c = sub.add_parser('build')
    c.add_argument('sessions', type=Path, nargs='+')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--base', type=Path, default=Path('models/yolo11n-seg.pt'))
    c.add_argument('--names', type=Path)
    c.add_argument('--allow-unreviewed-train', action='store_true',
                   help='train split may use teacher pseudo-labels; val/test always need reviewed labels')
    c = sub.add_parser('train')
    c.add_argument('--data', type=Path, required=True); c.add_argument('--output', type=Path, required=True)
    c.add_argument('--base', type=Path, default=Path('models/yolo11n-seg.pt'))
    c.add_argument('--epochs', type=int, default=30); c.add_argument('--imgsz', type=int, default=640)
    c.add_argument('--batch', type=int, default=8); c.add_argument('--device', default='cpu')
    c.add_argument('--lr0', type=float, default=0.001); c.add_argument('--freeze', type=int, default=10)
    a = p.parse_args(argv)
    dict(capture=capture, prelabel=prelabel, review=review, build=build, train=train)[a.command](a)


if __name__ == '__main__':
    main()
