"""Measure real local inference; synthetic smoke input is not a webcam accuracy test.

Run with --image only for approved adult/doll/object test images. No images/video are saved.
The default input is a generated, nonhuman geometric scene; report the input scope honestly.
"""
import argparse
import hashlib
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from childsafe.config import load_config
from childsafe.runtime import configure_runtime
from childsafe.pipeline import FramePipeline
from childsafe.vision.detection import SegmentationDetector
from childsafe.vision.object_classifier import ObjectClassifier
from childsafe.vision.landmarks import LandmarkDetector


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(times):
    return dict(median_s=float(np.median(times)),p90_s=float(np.percentile(times,90)),
                throughput_fps=len(times)/sum(times),samples=len(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector',required=True)
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--face-model',required=True)
    ap.add_argument('--hand-model',required=True)
    ap.add_argument('--image')
    ap.add_argument('--samples',type=int,default=10)
    ap.add_argument('--cpu-threads',type=int,default=2)
    ap.add_argument('--out',default='outputs/benchmark_runtime.json')
    args = ap.parse_args()
    if args.samples < 2:
        raise SystemExit('need at least two measured samples')
    if args.image:
        rgb = np.asarray(Image.open(args.image).convert('RGB'))
        input_kind = 'user-approved local image'
    else:
        img = Image.new('RGB',(640,480),(170,170,170))
        draw = ImageDraw.Draw(img)
        draw.ellipse((200,100,350,250),fill=(255,140,0))
        draw.rectangle((40,300,520,320),fill=(35,80,160))
        rgb = np.asarray(img)
        input_kind = 'generated nonhuman geometric scene; smoke test only'
    cfg = load_config()
    cfg['runtime']['cpu_threads']=args.cpu_threads
    cfg['detector']['weights']=args.detector
    configure_runtime(cfg)
    detector = SegmentationDetector(args.detector,device='cpu',cpu_threads=args.cpu_threads)
    if not detector.ok:
        raise SystemExit('detector initialization failed: '+detector.error)
    classifier = ObjectClassifier(args.ckpt,device='cpu')
    landmarks = LandmarkDetector(args.face_model,args.hand_model)
    cfg['paths'].update(face_model=args.face_model,hand_model=args.hand_model)
    pipe = FramePipeline(cfg,None,detector=detector,classifier=classifier)
    # CPU thread count is recorded; avoid oversubscription on small models.
    import torch
    measurements = {name:[] for name in ['detector','classifier','landmarks','pipeline']}
    last_obj = None
    last_detections = []
    try:
        for i in range(args.samples+2):
            for name,call in [('detector',lambda:detector.detect(rgb)),
                              ('classifier',lambda:classifier(rgb)),
                              ('landmarks',lambda:landmarks.detect(rgb,i/30)),
                              ('pipeline',lambda:pipe.process(rgb,i/30))]:
                start = time.perf_counter()
                result = call()
                elapsed = time.perf_counter()-start
                if i >= 2:
                    measurements[name].append(elapsed)
                if name == 'classifier': last_obj = result
                if name == 'detector': last_detections = result
                if name == 'detector' and not detector.ok:
                    raise RuntimeError('detector inference failed: '+detector.error)
    finally:
        landmarks.close(); pipe.close()
    report = dict(input_scope=input_kind,image_shape=list(rgb.shape),device='CPU',
                  cpu_threads=torch.get_num_threads(),platform=platform.platform(),
                  input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                  models={name:dict(sha256=checksum(path),file=Path(path).name) for name,path in
                          [('detector',args.detector),('classifier',args.ckpt),('face',args.face_model),('hand',args.hand_model)]},
                  versions={p:metadata.version(p) for p in ['numpy','Pillow','torch','torchvision','timm','ultralytics',
                                                           'mediapipe','opencv-python','pandas','PyYAML','pytest']},
                  timings={name:summary(times) for name,times in measurements.items()},
                  classifier_output=last_obj,
                  detector_output=[dict(label=d.label,confidence=d.confidence,box=list(d.box),
                                        mask_shape=list(d.mask.shape) if d.mask is not None else None,
                                        mask_pixels=int(d.mask.sum()) if d.mask is not None else 0)
                                   for d in last_detections],
                  limitations=['No event recall or false-alarm estimates from a single smoke-test image.',
                               'Empty-person scene does not benchmark four-person MediaPipe workload.',
                               'No webcam/elevated-view calibration. No end-to-end camera latency.'])
    dest = Path(args.out); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
