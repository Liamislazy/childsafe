"""Reproducible adult/doll numerical sessions; never opens cameras or stores media."""
import hashlib
import json
import re
from pathlib import Path


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def validate_sessions(sessions):
    seen, subjects = set(), {}
    for s in sessions:
        if s['recording_id'] in seen:
            raise ValueError('duplicate recording id')
        seen.add(s['recording_id'])
        if s['split'] not in ('dev','test') or s['subject_kind'] not in ('adult','doll'):
            raise ValueError('only dev/test adult/doll sessions are supported')
        if s['subject_kind'] == 'adult' and s.get('consent_confirmed') is not True:
            raise ValueError('adult consent must be explicitly confirmed')
        if not s['subjects'] or not s['condition']:
            raise ValueError('subjects and condition required')
        for subject in s['subjects']:
            if subject in subjects and subjects[subject] != s['split']:
                raise ValueError('subject appears in both dev and test')
            subjects[subject] = s['split']


def create_session(root, recording_id, split, subjects, condition, cfg, ckpt,
                   subject_kind='adult', consent_confirmed=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',recording_id):
        raise ValueError('recording id must be a plain slug')
    root = Path(root)
    meta = dict(version=1,recording_id=recording_id,split=split,subjects=subjects,
                subject_kind=subject_kind,consent_confirmed=consent_confirmed,condition=condition,
                config_sha256=config_hash(cfg),classifier_sha256=file_hash(ckpt),
                detector_sha256=file_hash(cfg['detector']['weights']),
                annotation_scope='camera',start_s=None,end_s=None,
                features='features.csv',annotations='annotations.csv',
                status='awaiting independent annotation and numerical features')
    previous = [json.loads(p.read_text(encoding='utf8')) for p in root.glob('*/manifest.json')]
    validate_sessions(previous+[meta])
    directory = root/recording_id
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    (directory/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf8')
    (directory/'annotations.csv').write_text('start,end,label\n',encoding='utf8')
    (directory/'observation_labels.csv').write_text('session_id,person_id,t,mouth_observable\n',encoding='utf8')
    return directory
