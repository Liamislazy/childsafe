"""Synthetic metadata tests, not clinical evidence or a trained choking detector."""
import json
import pandas as pd
import pytest
from childsafe.evaluation.choking import choking_report, PREDICTION_COLUMNS


def recording(kind='simulation'):
    return pd.DataFrame([dict(recording_id='r', duration_s=3., evidence_kind=kind,
                              age_group='adult_simulator', viewpoint='elevated')])


def annotations():
    return pd.DataFrame([
        dict(recording_id='r', subject_id='1', start=0., end=1., diagnosis='non_choking', cause='unknown', review_reference='sim-script'),
        dict(recording_id='r', subject_id='1', start=1., end=2., diagnosis='choking', cause='toy', review_reference='sim-script'),
        dict(recording_id='r', subject_id='1', start=2., end=3., diagnosis='uncertain', cause='unknown', review_reference='sim-script')])


def predictions():
    return pd.DataFrame([dict(recording_id='r', subject_id='1', t=i/10,
                              choking_alarm=int(1.2 <= i/10 < 1.8), choking_observation='observable')
                         for i in range(30)])


def test_simulation_is_never_reported_as_clinical_success():
    report = choking_report(predictions(), annotations(), recording())
    assert report['by_evidence_kind']['clinical'] is None
    sim = report['by_evidence_kind']['simulation']
    assert sim['event_recall'] == 1 and sim['toy_event_recall'] == 1
    assert sim['latency_median_s'] == pytest.approx(.2)
    assert sim['observation_coverage'] < 1  # no evidence beyond the last sample
    assert report['clinical_validity'] == 'not established by this metadata evaluator'
    json.dumps(report, allow_nan=False)


def test_missed_subject_and_missing_observation_remain_in_recall_denominator():
    refs = annotations()
    missed = refs[refs.diagnosis=='choking'].copy(); missed['subject_id']='2'
    frames = predictions(); frames['choking_observation']='unobservable'
    report = choking_report(frames, pd.concat([refs,missed]), recording())
    sim = report['by_evidence_kind']['simulation']
    assert sim['n_choking_events']==2 and sim['event_recall']==.5
    assert sim['observation_coverage']==0
    # Completely absent model output is not interpreted as a healthy patient.
    empty = pd.DataFrame(columns=sorted(PREDICTION_COLUMNS))
    report = choking_report(empty, refs, recording())
    assert report['by_evidence_kind']['simulation']['event_recall']==0


def test_unknown_reference_time_is_not_counted_as_negative():
    frames=predictions(); frames['choking_alarm']=0
    frames.loc[frames.t>=2.2,'choking_alarm']=1
    report=choking_report(frames,annotations(),recording())
    sim=report['by_evidence_kind']['simulation']
    assert sim['known_non_choking_s']==1
    assert sim['false_alarm_episodes']==0
    assert sim['unknown_reference_alarm_episodes']==1


def test_an_alarm_stuck_on_cannot_hide_false_alarms_behind_a_positive_event():
    frames=predictions(); frames['choking_alarm']=1
    report=choking_report(frames,annotations(),recording())
    sim=report['by_evidence_kind']['simulation']
    assert sim['event_recall']==1 and sim['false_alarm_episodes']==1


def test_clinical_references_require_review_and_are_separate_from_simulation():
    refs=annotations(); refs['review_reference']=''
    with pytest.raises(ValueError,match='review_reference'):
        choking_report(predictions(),refs,recording('clinical'))
    refs['review_reference']='synthetic-review-id'  # schema test only, not a clinical case
    simulated=recording(); clinical=recording('clinical'); clinical['recording_id']='clinical-fixture'
    clinical_refs=refs.copy(); clinical_refs['recording_id']='clinical-fixture'
    report=choking_report(predictions(),pd.concat([annotations(),clinical_refs]),pd.concat([simulated,clinical]))
    assert report['by_evidence_kind']['simulation']['event_recall']==1
    assert report['by_evidence_kind']['clinical']['event_recall']==0


def test_mouthing_levels_duplicate_cameras_and_overlapping_references_are_rejected():
    frames=predictions().rename(columns={'choking_alarm':'level'})
    with pytest.raises(ValueError,match='not a choking prediction'):
        choking_report(frames,annotations(),recording())
    frames=predictions()
    with pytest.raises(ValueError,match='fuse cameras'):
        choking_report(pd.concat([frames,frames.iloc[:1]]),annotations(),recording())
    refs=annotations(); refs.loc[1,'start']=.5
    with pytest.raises(ValueError,match='nonoverlapping'):
        choking_report(frames,refs,recording())


def test_long_frame_gaps_do_not_establish_observation_coverage():
    frames=predictions().iloc[[0,15]].copy()
    report=choking_report(frames,annotations(),recording())
    assert report['by_evidence_kind']['simulation']['observation_coverage']==0
