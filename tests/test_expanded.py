import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from childsafe.alerts.processor import AlertProcessor, CAUTION, NORMAL
from childsafe.classifier.engine import summarize
from childsafe.config import load_config
from childsafe.data.dataset import Letterbox
from childsafe.evaluation.replay import replay

def feature(t, valid=True):
    return {'t':t,'face_ok':int(valid),'hand_ok':int(valid),
            'd_face_width':.1 if valid else np.nan,'d_last_mouth':np.nan}

def test_low_category_confidence_never_becomes_safe_and_replays():
    cfg=load_config(); cfg['smoothing']['method']='none'
    obj={'p_unsafe_raw':.1,'p_unsafe_cal':.1,'obj_class':'fruit','obj_conf':.27}
    frames=[feature(i/30)|obj for i in range(45)]
    live=AlertProcessor(cfg)
    outputs=[live.step(f,obj if live.wants_classification(f) else None) for f in frames]
    assert outputs[-1]['obj_status']=='uncertain'
    assert outputs[-1]['level']==CAUTION
    offline=replay(pd.DataFrame(frames),cfg)
    assert offline.level.tolist()==[o['level'] for o in outputs]
    assert offline.obj_status.tolist()==[o['obj_status'] for o in outputs]

def test_legacy_other_is_unknown_even_if_mapped_safe():
    cfg=load_config(); p=AlertProcessor(cfg)
    out=p.step(feature(0),{'p_unsafe_raw':.01,'p_unsafe_cal':.01,'obj_class':'other','obj_conf':.99})
    assert out['obj_status']=='uncertain'

def test_hand_and_face_occlusion_holds_caution_but_expires():
    cfg=load_config(); p=AlertProcessor(cfg)
    obj={'p_unsafe_raw':.05,'p_unsafe_cal':.05,'obj_class':'spoon','obj_conf':.9}
    for i in range(30): p.step(feature(i/30),obj)
    out=p.step(feature(1.1,False))
    assert out['lost_near'] and out['level']>=CAUTION
    for i in range(90): out=p.step(feature(2+i/30,False))
    assert not out['lost_near'] and out['level']==NORMAL

def test_rejection_metrics_count_low_class_confidence_as_caution():
    # Three safe classes accumulate .90; no individual category is confident.
    logits=np.log([[.1,.3,.3,.3]])
    m=summarize(logits,np.array([0]),['pen','fruit','spoon','toy'],
                {'pen':'unsafe','fruit':'safe','spoon':'safe','toy':'safe'},reject_threshold=.7)
    assert m['binary']['fn']==1
    assert m['reject']['unsafe_missed_as_safe_confidently']==0
    assert m['reject']['unsafe_rejected_to_caution']==1

def test_letterbox_keeps_endpoints_of_thin_objects():
    a=np.zeros((20,200,3),dtype=np.uint8); a[:,:8]=[255,0,0]; a[:,-8:]=[0,255,0]
    b=np.array(Letterbox(224)(Image.fromarray(a)))
    assert b.shape==(224,224,3)
    assert ((b[:,:,0]>200)&(b[:,:,1]<30)).any()
    assert ((b[:,:,1]>200)&(b[:,:,0]<30)).any()

def load_prepare():
    spec=importlib.util.spec_from_file_location('prepare',Path(__file__).resolve().parents[1]/'scripts/prepare_expanded_data.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def test_manifest_blocks_unreviewed_data_and_keeps_source_groups(tmp_path):
    base=[]
    rng=np.random.default_rng(7)
    for i,role in enumerate(['train','val','val','test']):
        path=tmp_path/f'base{i}.jpg'; Image.fromarray(rng.integers(0,256,(24,24,3),dtype=np.uint8)).save(path)
        base.append({'path':path.name,'category':'spoon','label':'safe','group':str(i),'split_group':role})
    extra=[]
    for i in range(12):
        path=tmp_path/f'pen{i}.jpg'; Image.fromarray(rng.integers(0,256,(24,24,3),dtype=np.uint8)).save(path)
        extra.append({'path':path.name,'category':'pen','label':'unsafe','group':f'source{i}',
                      'domain':'adult_simulation','reviewed':False})
    manifest=tmp_path/'manifest.csv'; pd.DataFrame(extra).to_csv(manifest,index=False)
    module=load_prepare()
    with pytest.raises(ValueError,match='Review manifest'): module.merge_manifest(pd.DataFrame(base),tmp_path,manifest,audit=False)
    for r in extra: r['reviewed']=True
    # A second crop from the same source must stay in the same partition.
    extra.append(extra[0].copy()); pd.DataFrame(extra).to_csv(manifest,index=False)
    df,_=module.merge_manifest(pd.DataFrame(base),tmp_path,manifest,audit=False)
    assert df.groupby('group').split_group.nunique().max()==1
    assert df.loc[df.path==str(tmp_path/'base3.jpg'),'split_group'].item()=='test'
    assert set(df[df.category=='pen'].split_group)=={'train','val','cal','test'}
