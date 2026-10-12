import pandas as pd
import pytest

from childsafe.data.audio_labels import audit_coughvid_metadata


def test_acoustic_annotations_are_not_clinical_events_or_silent_negatives():
    df = pd.DataFrame([
        dict(uuid="a", age=8, choking_1="True", quality_1="good", choking_2="False", quality_2="ok"),
        dict(uuid="b", age=30, choking_1=None, quality_1=None, choking_2=None, quality_2=None),
        dict(uuid="c", age=None, choking_1="True", quality_1="poor", choking_2=None, quality_2=None),
        dict(uuid="d", age=50, choking_1="False", quality_1="no_cough", choking_2=None, quality_2=None),
    ])
    report = audit_coughvid_metadata(df)
    c = report["counts"]
    assert c["any_positive"] == 2 and c["usable_any_positive"] == 1
    assert c["reviewer_disagreement"] == 1 and c["usable_reviewer_disagreement"] == 1
    assert c["at_least_two_reviewers_all_positive"] == 0 and c["single_reviewer_positive"] == 1
    assert c["unannotated_recordings"] == 1 and c["explicitly_negative_only"] == 1
    assert report["confirmed_pediatric_toy_obstruction_events"] is None
    assert not report["subject_independent_split_possible_from_uuid"]
    assert report["by_self_reported_age"]["under_18_self_reported"]["any_positive"] == 1
    assert report["per_reviewer"]["1"]["positive"] == 2


@pytest.mark.parametrize("change", [dict(choking_1="unknown"), dict(uuid=""),
                                    dict(age=-1), dict(age="bad"), dict(quality_1="great")])
def test_invalid_or_ambiguous_schema_fails(change):
    row = dict(uuid="a", age=30, choking_1="True", quality_1="good")
    row.update(change)
    with pytest.raises(ValueError):
        audit_coughvid_metadata(pd.DataFrame([row]))


def test_missing_expert_columns_and_duplicate_uuid_fail():
    with pytest.raises(ValueError):
        audit_coughvid_metadata(pd.DataFrame([dict(uuid="a", age=20, status_SSL="healthy")]))
    row = dict(uuid="a", age=20, choking_1=False, quality_1="good")
    with pytest.raises(ValueError):
        audit_coughvid_metadata(pd.DataFrame([row, row]))
