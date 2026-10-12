"""Strict adult/doll event reports with explicitly labeled negative exposure."""
import math
import numpy as np
from scipy.stats import chi2
from childsafe.alerts.temporal import frames_to_events
from childsafe.evaluation.replay import replay,aggregate_results
from childsafe.evaluation.runtime_profile import runtime_profile


def summarize(counts):
    n,hit,fa,hours=counts['events'],counts['hits'],counts['false_alarms'],counts['negative_hours']
    lat=counts['latencies']
    if n:
        p=hit/n; z=1.959963984540054; den=1+z*z/n
        center=(p+z*z/(2*n))/den
        margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        ci=[max(0.,center-margin),min(1.,center+margin)]
    else: ci=None
    return dict(**counts,event_recall=hit/n if n else None,recall_wilson_95=ci,
                false_alarms_per_hour=fa/hours if hours else None,
                fa_poisson_95=([float(chi2.ppf(.025,2*fa)/(2*hours)) if fa else 0.,
                                float(chi2.ppf(.975,2*(fa+1))/(2*hours))] if hours else None),
                latency_median_s=float(np.median(lat)) if lat else None,
                latency_p90_s=float(np.percentile(lat,90)) if lat else None,
                observable_event_recall=counts['observable_hits']/counts['observable_events'] if counts['observable_events'] else None,
                observation_coverage=counts['observable_s']/counts['duration_s'] if counts['duration_s'] and not counts['legacy_quality'] else None,
                interval_note='95% intervals assume independent events/Poisson alarms; repeated scenes may violate this')


def evaluate_session(features, annotations, cfg, start, end, scope='camera', baseline=False,
                     components=None,min_level=2):
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        raise ValueError('explicit finite session start/end required')
    if scope not in ('camera','person') or (scope=='person' and 'person_id' not in annotations):
        raise ValueError('invalid annotation scope')
    if not set(annotations.label) <= {'unsafe_mouthing','non_event','uncertain'}:
        raise ValueError('use explicit positive/non_event/uncertain labels')
    if not np.isfinite(annotations[['start','end']].to_numpy(float)).all():
        raise ValueError('finite annotation times required')
    if not features.empty and features.duplicated(['session_id','person_id','t']).any():
        raise ValueError('duplicate person sample')
    if not features.empty and features.session_id.nunique() != 1:
        raise ValueError('one runtime epoch per session manifest required')
    if ((annotations.start < start)|(annotations.end > end)|(annotations.start >= annotations.end)).any():
        raise ValueError('annotations outside declared session or invalid intervals')
    counts=dict(events=0,hits=0,false_alarms=0,unknown_alarm_episodes=0,negative_hours=0.,latencies=[],
                observable_events=0,observable_hits=0,observable_s=0.,duration_s=0.,legacy_quality=False)
    out=replay(features,cfg,components) if not features.empty else None
    if out is not None:
        counts['legacy_quality'] = any('legacy:' in str(r) for r in out.observation_reasons)
    if baseline and out is not None:
        from childsafe.evaluation.events import distance_only_levels
        field='d_'+cfg['landmarks']['normalizer']
        out['level']=distance_only_levels(features[field].to_numpy(),cfg['temporal']['d_enter'])
    if scope=='camera':
        groups=[(aggregate_results(out) if out is not None else None,annotations)]
    else:
        ids=set(annotations.person_id)|(set(out.person_id)-{0} if out is not None else set())
        groups=[(out[out.person_id==pid] if out is not None else None,annotations[annotations.person_id==pid]) for pid in ids]
    max_gap=cfg['temporal']['max_gap_s']
    for group,ann in groups:
        ordered=ann.sort_values('start')
        if len(ordered)>1 and (ordered.start.to_numpy()[1:] < ordered.end.to_numpy()[:-1]).any():
            raise ValueError('overlapping ground truth intervals')
        t=group.t.to_numpy(float) if group is not None else np.array([])
        active=group.level.to_numpy()>=min_level if group is not None else np.array([],bool)
        if len(t) and (not np.isfinite(t).all() or (np.diff(t)<=0).any() or t.min()<start or t.max()>end):
            raise ValueError('invalid or duplicated session timestamps')
        observed=group.observation.to_numpy()=='observable' if group is not None else np.array([],bool)
        if group is not None and 'observation_reasons' in group:
            counts['legacy_quality'] |= any('legacy:' in str(r) for r in group.observation_reasons)
        counts['duration_s']+=end-start
        if len(t)>1:
            dt=np.diff(t)
            counts['observable_s']+=float(dt[(dt<=max_gap)&observed[:-1]&observed[1:]].sum())
        positive=ann[ann.label=='unsafe_mouthing']
        negative=ann[ann.label=='non_event']
        counts['negative_hours']+=float((negative.end-negative.start).sum()/3600)
        pos_mask=np.zeros(len(t),bool); neg_mask=np.zeros(len(t),bool)
        for r in positive.itertuples():
            window=(t>=r.start)&(t<r.end); pos_mask |= window
            detected=active&window
            counts['events']+=1; counts['hits']+=int(detected.any())
            if detected.any(): counts['latencies'].append(float(t[detected][0]-r.start))
            inside=t[window]
            fully=(len(inside)>=2 and observed[window].all() and
                   np.diff(inside).max(initial=0)<=max_gap and
                   inside[0]-r.start<=max_gap and r.end-inside[-1]<=max_gap)
            counts['observable_events']+=int(fully)
            counts['observable_hits']+=int(fully and detected.any())
        for r in negative.itertuples(): neg_mask |= (t>=r.start)&(t<r.end)
        for s,e in frames_to_events(t,active,cfg['temporal']['merge_s']):
            episode=active&(t>=s)&(t<=e)
            # A sustained alarm cannot hide negative-time alerts by also hitting a positive.
            if (episode&neg_mask).any(): counts['false_alarms']+=1
            elif not (episode&pos_mask).any(): counts['unknown_alarm_episodes']+=1
    report=summarize(counts)
    report.update(annotation_scope=scope,min_level=min_level,performance=runtime_profile(features,max_gap),
                  observation_note='legacy quality unknown' if counts['legacy_quality'] else 'full declared duration; missing samples/persons unobserved',
                  target='simulated object-near-mouth risk, not choking or physical contact')
    return report
