"""Dev-only proposals from independent visibility labels; no automatic certification."""
import copy
import itertools
import numpy as np


def propose_quality(features, labels, cfg, split):
    if split != 'dev':
        raise ValueError('observation thresholds may only be proposed on dev')
    keys = ['session_id','person_id','t']
    if features.duplicated(keys).any() or labels.duplicated(keys).any():
        raise ValueError('duplicate annotation/sample')
    target = labels.mouth_observable
    if target.isna().any() or not target.isin([0,1,True,False]).all():
        raise ValueError('explicit independent visibility labels required')
    data = labels.merge(features,on=keys,how='left',validate='one_to_one',indicator=True)
    if not (data['_merge'] == 'both').all() or len(data) == 0:
        raise ValueError('each label must match a numerical feature sample')
    fields = ['original_face_px','original_mouth_px','head_yaw','head_pitch','brightness','sharpness']
    values = data[fields].to_numpy(float)
    if not data.quality_schema_version.eq(3).all():
        raise ValueError('calibration requires original-pixel quality schema v3')
    truth = data.mouth_observable.to_numpy(bool)
    if not truth.any() or truth.all():
        raise ValueError('both visible and nonvisible dev examples required')
    q = cfg['quality']; candidates = []
    for face,sharp in itertools.product(sorted({q['min_face_px'],30,40,60}),
                                        sorted({q['min_sharpness'],10,20,40})):
        prediction = (np.isfinite(values).all(1) & (values[:,0] >= face) &
                      (values[:,1] >= q['min_mouth_px']) & (abs(values[:,2]) <= q['max_yaw_deg']) &
                      (abs(values[:,3]) <= q['max_pitch_deg']) &
                      (values[:,4] >= q['min_brightness']) & (values[:,4] <= q['max_brightness']) &
                      (values[:,5] >= sharp))
        candidates.append(dict(min_face_px=face,min_sharpness=sharp,
                               false_visible=int((prediction & ~truth).sum()),
                               missed_visible=int((~prediction & truth).sum())))
    best = min(candidates,key=lambda c:(c['false_visible'],c['missed_visible'],
                                        abs(c['min_face_px']-q['min_face_px'])+abs(c['min_sharpness']-q['min_sharpness'])))
    proposed = copy.deepcopy(cfg)
    proposed['quality'].update(min_face_px=best['min_face_px'],min_sharpness=best['min_sharpness'],calibrated=False)
    return proposed, dict(split='dev',samples=len(data),selected=best,candidates=candidates,
                          scope='face size and sharpness proposal only; jitter/occlusion not calibrated',
                          certification='not established; calibrated remains false')
