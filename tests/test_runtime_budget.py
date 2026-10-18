from types import SimpleNamespace
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from childsafe.config import load_config
from childsafe.pipeline import FramePipeline,SceneLandmarks
from childsafe.vision.tracking import Track
from childsafe.evaluation.runtime_profile import runtime_profile


def test_optional_roi_work_is_bounded_and_fair_to_four_people():
    class Detector:
        ok=True; supports_roi=True
        def __init__(self): self.seen=[]
        def detect(self,rgb): self.seen.append(int(rgb[0,0,0])); return []
    detector=Detector(); cfg=load_config(); cfg['detector']['roi_min_px']=64
    pipe=FramePipeline(cfg,None,detector=detector)
    rgb=np.zeros((400,800,3),np.uint8)
    scene=SceneLandmarks(); tracks=[]
    for i in range(4):
        rgb[:,i*200:(i+1)*200]=i+1; x=i*200+100
        hand=np.column_stack((np.linspace(x-5,x+5,21),np.linspace(100,110,21),np.zeros(21)))
        tracks.append(Track(i+1,(i*200,0,(i+1)*200,400),0))
        scene.people[i+1]=SimpleNamespace(face=None,hands=[hand])
    for _ in range(8):
        pipe.roi_passes=0
        pipe._zoom_detect(rgb,scene,tracks,set())
        assert pipe.roi_passes==1
    assert detector.seen==[1,2,3,4]*2
    pipe.roi_passes=0
    pipe._zoom_detect(rgb,scene,tracks,set(),deadline=-1)
    assert pipe.roi_passes==0


def test_expired_detector_work_does_not_start_landmark_models(monkeypatch):
    from childsafe.vision.detection import Detection
    now=[0.]
    monkeypatch.setattr('childsafe.pipeline.time',SimpleNamespace(perf_counter=lambda:now[0]))
    class Detector:
        ok=True
        def detect(self,rgb):
            now[0]=1.
            return [Detection((0,0,100,100),'person',.99)]
    def forbidden(): raise AssertionError('expired work should not initialize MediaPipe')
    pipe=FramePipeline(load_config(),None,detector=Detector(),landmark_factory=forbidden)
    try:
        _,_,out=pipe.process(np.zeros((100,100,3),np.uint8),0,processing_deadline=.25)
        assert not out['green'] and not out['feature_rows'][0]['frame_valid']
        assert not next(iter(pipe.states.values()))['landmark_failed']
    finally: pipe.close()


def test_profile_counts_real_frames_and_budget_failures():
    rows=[dict(t=t,session_id=1,person_id=p,tracked=True,frame_valid=valid,
               inference_s=cost,detector_calls=2,roi_detector_calls=1)
          for t,cost,valid in [(0,.1,True),(.1,.3,False)] for p in (1,2)]
    result=runtime_profile(pd.DataFrame(rows))
    assert result['samples']==2 and result['detector_calls_total']==4
    assert result['roi_detector_calls_total']==2 and result['inference_budget_exceeded_samples']==1
    assert result['invalid_samples']==1 and result['samples_by_tracked_people']=={'2':2}


def test_live_profile_uses_real_elapsed_time_and_resets_on_disconnect():
    path=Path(__file__).resolve().parents[1]/'scripts/profile_live_session.py'
    spec=importlib.util.spec_from_file_location('live_profile',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    now=[0.]; calls=[]
    class Source:
        slot=SimpleNamespace(overwrites=2)
        def read(self,timeout):
            now[0]+=.1
            return SimpleNamespace(source_ok=now[0]<.15,session_id=1,received_t=now[0],frame=np.zeros((20,20,3),np.uint8))
    class Pipe:
        cfg=load_config()
        def reset(self): calls.append('reset')
        def process(self,rgb,t,**kwargs):
            return None,None,dict(feature_rows=[dict(t=t,frame_valid=True)])
    data,elapsed=mod.collect(Source(),Pipe(),.25,clock=lambda:now[0])
    assert elapsed>=.25 and len(data)==1 and calls.count('reset')==3
