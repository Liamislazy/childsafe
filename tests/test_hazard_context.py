import pandas as pd
import pytest
from childsafe.alerts.processor import AlertProcessor
from childsafe.config import load_config
from childsafe.evaluation.replay import replay
from test_observation import f


@pytest.mark.parametrize('category',['toy','fruit','spoon','ball','babyitems'])
def test_dataset_safe_category_near_mouth_requests_early_caution(category):
    cfg=load_config(); proc=AlertProcessor(cfg); rows=[]
    obj=dict(p_unsafe_raw=.01,p_unsafe_cal=.01,obj_conf=.99,obj_class=category,p_unknown_cal=0.)
    for i in range(15):
        row=f(i*.1); row['object_present']=True
        out=proc.step(row,obj); rows.append(dict(row,**obj))
    assert out['level']==1 and out['hazard_status']=='context_unknown'
    assert out['hazard_type']=='unknown' and not out['green']
    assert replay(pd.DataFrame(rows),cfg).level.tolist()==[0]+[1]*14


@pytest.mark.parametrize('category,expected',[('knife',2),('fork',2),('socket',1),('electricheater',1),('other',1)])
def test_high_alert_requires_defined_context_hazard_not_just_dataset_label(category,expected):
    proc=AlertProcessor(load_config())
    obj=dict(p_unsafe_raw=.99,p_unsafe_cal=.99,obj_conf=.99,obj_class=category,p_unknown_cal=0.)
    for i in range(20): out=proc.step(f(i*.1),obj)
    assert out['level']==expected


def test_weak_localization_and_unsupported_object_never_create_high_alert():
    proc=AlertProcessor(load_config())
    obj=dict(p_unsafe_raw=.01,p_unsafe_cal=.01,obj_conf=.99,obj_class='toy')
    for i in range(20):
        row=f(i*.1); row.update(proximity_evidence=False)
        out=proc.step(row,obj)
        assert out['level']<=1
    assert out['level']==1
