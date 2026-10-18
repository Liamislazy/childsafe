"""Measure full local camera processing; numerical logs only, no frame recording."""
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import pandas as pd

from childsafe.config import load_config
from childsafe.pipeline import FramePipeline
from childsafe.runtime import configure_runtime
from childsafe.runtime.source import VideoSource
from childsafe.evaluation.runtime_profile import runtime_profile


def collect(source,pipe,duration_s,clock=time.perf_counter):
    """Full-duration measurements include cold start, missing frames and reconnects."""
    start=clock(); end=start+duration_s; rows=[]; epoch=None
    while clock()<end:
        packet=source.read(.1)
        now=clock()
        if packet is None or not packet.source_ok:
            if packet is not None:
                pipe.reset(); epoch=None
            continue  # full requested duration remains the report denominator
        if epoch!=packet.session_id:
            pipe.reset(); epoch=packet.session_id
        rgb=cv2.cvtColor(packet.frame,cv2.COLOR_BGR2RGB)
        _,_,result=pipe.process(rgb,packet.received_t-start,received_t=packet.received_t,
                               frame_age_s=max(0.,now-packet.received_t),
                               processing_deadline=packet.received_t+pipe.cfg['runtime']['stale_frame_s'])
        for row in result['feature_rows']:
            rows.append(dict(row,capture_slot_overwrites=source.slot.overwrites))
    return pd.DataFrame(rows),clock()-start


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source_args=p.add_mutually_exclusive_group(required=True)
    source_args.add_argument('--source',type=int,help='local camera index')
    source_args.add_argument('--source-env',help='environment variable containing the stream URL; never exported')
    p.add_argument('--kind',choices=['adult','doll'],required=True)
    p.add_argument('--consent-confirmed',action='store_true')
    p.add_argument('--condition',required=True)
    p.add_argument('--config'); p.add_argument('--ckpt',required=True)
    p.add_argument('--seconds',type=float,default=60.)
    p.add_argument('--width',type=int,default=1280); p.add_argument('--height',type=int,default=720)
    p.add_argument('--max-people',type=int,choices=[1,2,4],default=4)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--out',type=Path,default=Path('outputs/live_profile'))
    a=p.parse_args(argv)
    source_value=a.source if a.source is not None else os.environ.get(a.source_env)
    if source_value is None or source_value=='': p.error('source environment variable is empty')
    if a.kind=='adult' and not a.consent_confirmed: p.error('adult participants must have confirmed consent')
    if a.seconds<=0 or not a.seconds<3601 or min(a.width,a.height)<=0: p.error('invalid duration or dimensions')
    cfg=load_config(a.config)
    cfg['runtime'].update(width=a.width,height=a.height,max_people=a.max_people,cpu_threads=a.cpu_threads)
    configure_runtime(cfg)
    pipe=FramePipeline(cfg,a.ckpt)
    if pipe.classifier is None or not pipe.detector.ok:
        pipe.close(); p.error('local classifier/detector unavailable; no performance certification')
    source=VideoSource(source_value,a.width,a.height)
    try:
        source.start(); data,elapsed=collect(source,pipe,a.seconds)
    finally:
        source.close(); pipe.close()
    report=runtime_profile(data,cfg['temporal']['max_gap_s'])
    import torch
    report.update(condition=a.condition,subject_kind=a.kind,consent_confirmed=a.consent_confirmed,
                  duration_s=elapsed,config=cfg,provenance=pipe.provenance,
                  torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),
                  measured_device=str(pipe.classifier.device),
                  scope='full runtime timing; no independently labeled event accuracy or sensor capture latency')
    a.out.mkdir(parents=True,exist_ok=True)
    data.to_csv(a.out/'features.csv',index=False)
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(a.out/'report.json')


if __name__=='__main__': main()
