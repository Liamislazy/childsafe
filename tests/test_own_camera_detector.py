import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/own_camera_detector.py'
    spec=importlib.util.spec_from_file_location('own_camera_detector',path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def names_file(tmp_path):
    path=tmp_path/'names.yaml'; path.write_text('names: [person, fork, teddy bear]\n',encoding='utf-8')
    return path


def session(root,name,split,labels,participant='P01',review=None,pseudo=None,content=None):
    """labels: {stem: rows or None}; reviewed labels only for stems given in labels."""
    s=root/name; (s/'images').mkdir(parents=True)
    stems=sorted(set(labels)|set(pseudo or {})|set(review or {}))
    for stem in stems:
        (s/'images'/f'{stem}.jpg').write_bytes((content or {}).get(stem,f'{name}-{stem}'.encode()))
    (s/'labels_reviewed').mkdir()
    for stem,rows in labels.items():
        (s/'labels_reviewed'/f'{stem}.txt').write_text(''.join(r+'\n' for r in rows),encoding='utf-8')
    if pseudo:
        (s/'labels_prelabel').mkdir()
        for stem,rows in pseudo.items():
            (s/'labels_prelabel'/f'{stem}.txt').write_text(''.join(r+'\n' for r in rows),encoding='utf-8')
    if review: (s/'review.json').write_text(json.dumps(review),encoding='utf-8')
    (s/'session.json').write_text(json.dumps(dict(session=name,split=split,participant=participant,
                                  conditions=['dim'],consent_confirmed=True)),encoding='utf-8')
    return s


ROW='1 0.1 0.1 0.2 0.1 0.2 0.2'


def args(sessions,out,names,allow=False):
    return SimpleNamespace(sessions=sessions,out=out,names=names,base=None,allow_unreviewed_train=allow)


def test_polygon_line_validates_points():
    m=module()
    assert m.polygon_line(1.0,[[0,0],[1,0],[.5,.25]])=='1 0.000000 0.000000 1.000000 0.000000 0.500000 0.250000'
    with pytest.raises(ValueError): m.polygon_line(0,[[0,0],[1,1]])
    with pytest.raises(ValueError): m.polygon_line(0,[[0,0],[1.2,0],[0,1]])


def test_load_names_accepts_list_or_mapping(tmp_path):
    m=module(); path=tmp_path/'n.yaml'
    path.write_text('names: {0: person, 2: fork}\n',encoding='utf-8')
    assert m.load_names(path)=={0:'person',2:'fork'}
    assert m.load_names(names_file(tmp_path))=={0:'person',1:'fork',2:'teddy bear'}


def test_build_keeps_sessions_whole_and_never_tests_on_pseudo_labels(tmp_path):
    m=module(); root=tmp_path/'rec'
    sessions=[session(root,'tr1','train',{'a':[ROW]},pseudo={'b':[ROW],'c':[ROW]},review={'c':'needs_fix'}),
              session(root,'va1','val',{'a':[]},participant='P02'),
              session(root,'te1','test',{'a':[ROW]},pseudo={'b':[ROW]},participant='P01')]
    out=tmp_path/'ds'; m.build(args(sessions,out,names_file(tmp_path),allow=True))
    report=json.loads((out/'report.json').read_text())
    # train: reviewed a + pseudo b; c was flagged wrong. test: pseudo b is never used.
    assert report['frames_used']=={'train':2,'val':1,'test':1}
    assert report['label_status']['train']['needs_fix']==1
    assert report['label_status']['test']['unreviewed']==1
    assert report['participants_shared_between_train_and_test']==['P01']
    assert (out/'labels'/'val'/'a.txt').read_text()==''   # negative frame kept
    assert sorted(p.name for p in (out/'images'/'train').iterdir())==['a.jpg','b.jpg']


def test_build_without_flag_ignores_pseudo_labels_and_requires_every_split(tmp_path):
    m=module(); root=tmp_path/'rec'
    sessions=[session(root,'tr1','train',{},pseudo={'a':[ROW]}),
              session(root,'va1','val',{'a':[ROW]}),session(root,'te1','test',{'a':[ROW]})]
    with pytest.raises(ValueError,match='empty split'): m.build(args(sessions,tmp_path/'ds',names_file(tmp_path)))


def test_build_rejects_duplicate_frames_across_splits_and_bad_rows(tmp_path):
    m=module(); root=tmp_path/'rec'; same={'a':b'same-bytes'}
    sessions=[session(root,'tr1','train',{'a':[ROW]},content=same),
              session(root,'va1','val',{'a':[ROW]}),session(root,'te1','test',{'a':[ROW]},content=same)]
    with pytest.raises(ValueError,match='crosses splits'): m.build(args(sessions,tmp_path/'ds',names_file(tmp_path)))
    root2=tmp_path/'rec2'
    bad=[session(root2,'tr1','train',{'a':['7 0.1 0.1 0.2 0.1 0.2 0.2']}),
         session(root2,'va1','val',{'a':[ROW]}),session(root2,'te1','test',{'a':[ROW]})]
    with pytest.raises(ValueError,match='bad row'): m.build(args(bad,tmp_path/'ds2',names_file(tmp_path)))


def test_build_refuses_sessions_without_consent(tmp_path):
    m=module(); s=session(tmp_path/'rec','tr1','train',{'a':[ROW]})
    meta=json.loads((s/'session.json').read_text()); meta['consent_confirmed']=False
    (s/'session.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='consent'): m.build(args([s],tmp_path/'ds',names_file(tmp_path)))


def test_capture_requires_consent_flag(tmp_path):
    m=module()
    with pytest.raises(SystemExit): m.main(['capture','--session','s1','--split','train','--participant','P01',
                                            '--root',str(tmp_path)])
